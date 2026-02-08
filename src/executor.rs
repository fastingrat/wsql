use crate::{gpu::Gpu, jit};

pub struct QueryExecutor {
    gpu: Gpu,
}

impl QueryExecutor {
    pub fn new(gpu: Gpu) -> Self {
        Self { gpu }
    }

    pub async fn execute_batch(
        &self,
        batch: &arrow::record_batch::RecordBatch,
        projection: &jit::Expression,
        filter: Option<&jit::Expression>,
    ) -> anyhow::Result<Vec<i32>> {
        let mut used_cols = std::collections::BTreeSet::new();
        // Columns in the query
        jit::collect_columns(projection, &mut used_cols);

        if let Some(f) = filter {
            jit::collect_columns(f, &mut used_cols);
        }

        // Map columns to sequential bindings
        let mut mapping = std::collections::BTreeMap::new();
        for (idx, &col) in used_cols.iter().enumerate() {
            mapping.insert(col, idx as u32);
        }

        // Generate shader from mapping
        let wgsl = jit::generate_fused_shader(projection, filter, &mapping);

        // LOAD SHADER
        let shader = self
            .gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Dynamic Shader"),
                source: wgpu::ShaderSource::Wgsl(wgsl.into()),
            });

        // PIPELINE
        let pipeline = self
            .gpu
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Compute Pipeline"),
                layout: None,
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // BUFFERS
        let row_count = batch.num_rows() as u32;
        let size = (row_count * 4) as u64;
        let workgroup_count = row_count.div_ceil(64);
        let mut input_buffers = Vec::new();
        let output_buffer = self.gpu.output_buffer("out", size);
        let stagging_buffer = self.gpu.stagging_buffer("stage", size);
        let uniform_buffer = self.gpu.metadata_buffer("params", row_count);

        // BIND GROUP
        // Fill Input buffers
        for &col_idx in &used_cols {
            let data = batch
                .column(col_idx as usize)
                .as_any()
                .downcast_ref::<arrow::array::Int32Array>()
                .ok_or_else(|| anyhow::anyhow!("Expected Int32Array"))?;

            input_buffers.push(self.gpu.input_buffer("col", data.values()));
        }

        // bind group entries from input buffers
        let mut entries = Vec::new();

        for (binding_idx, buf) in input_buffers.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: binding_idx as u32,
                resource: buf.as_entire_binding(),
            });
        }

        let out_idx = used_cols.len() as u32;
        // output buffer
        entries.push(wgpu::BindGroupEntry {
            binding: out_idx,
            resource: output_buffer.as_entire_binding(),
        });
        // unifrom buffer ** at last
        entries.push(wgpu::BindGroupEntry {
            binding: out_idx + 1,
            resource: uniform_buffer.as_entire_binding(),
        });

        // create bind group
        let bind_group = self
            .gpu
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Dynamic Bind Group"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &entries,
            });

        // EXECUTE
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        // COPY TO STAGING BUFFER
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &stagging_buffer, 0, size);
        self.gpu.queue.submit(Some(encoder.finish()));

        // BACK TO CPU
        let buffer_slice = stagging_buffer.slice(..);
        let (sender, receiver) = tokio::sync::oneshot::channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
            let _ = sender.send(v);
        });

        let _ = self.gpu.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        // wait for gpu
        receiver
            .await
            .map_err(|e| anyhow::anyhow!("Channel error: {e}"))??;

        let data = buffer_slice.get_mapped_range();
        let result = bytemuck::cast_slice(&data).to_vec();
        Ok(result)
    }
}
