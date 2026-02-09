use crate::{gpu::Gpu, jit, sub::PhysicalPlan};

pub struct QueryExecutor {
    gpu: Gpu,
}

#[derive(Debug, PartialEq)]
pub enum QueryResult {
    Projection(Vec<i32>),
    Aggregate(f32),
}

impl QueryExecutor {
    pub fn new(gpu: Gpu) -> Self {
        Self { gpu }
    }

    pub async fn execute(
        &self,
        batch: &arrow::record_batch::RecordBatch,
        physical_plan: &PhysicalPlan,
    ) -> anyhow::Result<QueryResult> {
        let mut used_cols = std::collections::BTreeSet::new();
        // Columns in the query
        jit::collect_columns(&physical_plan.projection, &mut used_cols);

        // Check for filters
        if let Some(f) = &physical_plan.filter {
            jit::collect_columns(f, &mut used_cols);
        }

        // Map columns to sequential bindings
        let mut mapping = std::collections::BTreeMap::new();
        for (idx, &col) in used_cols.iter().enumerate() {
            mapping.insert(col, idx as u32);
        }

        // Generate shader from mapping
        let wgsl = jit::generate_shader(physical_plan, &mapping);
        if cfg!(debug_assertions) {
            wgsl.lines()
                .enumerate()
                .for_each(|(i, l)| println!("{:>3} | {}", i + 1, l));
        }
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
        let workgroup_count = row_count.div_ceil(64);
        let output_len = if physical_plan.is_aggregate {
            // aggregtion 6400 rows need 100 write operation
            workgroup_count
        } else {
            // projection 6400 rows needs 6400 write operations
            row_count
        };
        let size = (output_len * 4) as u64;
        let mut input_buffers: Vec<wgpu::Buffer> = Vec::new();
        let output_buffer = self.gpu.output_buffer("out", size);
        let stagging_buffer = self.gpu.stagging_buffer("stage", size);
        let uniform_buffer = self.gpu.metadata_buffer("params", row_count);

        // BIND GROUP
        // Fill Input buffers
        for &col_idx in &used_cols {
            let data = batch.column(col_idx as usize);

            match data.data_type() {
                arrow::datatypes::DataType::Int32 => {
                    let array = data
                        .as_any()
                        .downcast_ref::<arrow::array::PrimitiveArray<arrow::datatypes::Int32Type>>()
                        .unwrap();
                    input_buffers.push(self.gpu.input_buffer("col", array.values()));
                }
                arrow::datatypes::DataType::Float32 => {
                    let array = data.as_any().downcast_ref::<arrow::array::PrimitiveArray<arrow::datatypes::Float32Type>>().unwrap();
                    input_buffers.push(self.gpu.input_buffer("col", array.values()));
                }
                _ => anyhow::bail!("Unsupported datatype"),
            };
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

        println!("Submitting command encoder");
        self.gpu.queue.submit(Some(encoder.finish()));

        // BACK TO CPU
        println!("Mapping Buffer");
        let buffer_slice = stagging_buffer.slice(..);
        let (sender, receiver) = tokio::sync::oneshot::channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
            let _ = sender.send(v);
        });

        println!("Polling");
        self.gpu
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|e| anyhow::anyhow!("GPU Poll error: {e}"))?;

        // wait for gpu
        // Handle aggregate
        println!("Awaiting receiver");
        let result_val = receiver
            .await
            .map_err(|_| anyhow::anyhow!("Channel closed"))?;
        result_val.map_err(|e| anyhow::anyhow!("Buffer mapping failed: {e}"))?;

        let data = buffer_slice.get_mapped_range();

        let final_result = if physical_plan.is_aggregate {
            let partials: &[f32] = bytemuck::cast_slice(&data);
            let actual_partials = &partials[0..workgroup_count as usize];
            QueryResult::Aggregate(actual_partials.iter().sum())
        } else {
            let mut result = bytemuck::cast_slice(&data).to_vec();
            result.truncate(row_count as usize);
            QueryResult::Projection(result)
        };

        drop(data);
        stagging_buffer.unmap();
        Ok(final_result)
    }
}
