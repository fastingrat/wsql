use arrow::array::AsArray;
use std::borrow::Cow;

mod gpu;
mod jit;
mod sub;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // GPU
    let gpu = gpu::Gpu::new().await;

    // DATA
    let mut dal_builder = opendal::services::Fs::default().root(".");
    let dal_op = opendal::Operator::new(dal_builder)?.finish();
    let dal_buffer = dal_op.read("data/alltypes_plain.parquet").await?;
    let dal_bytes = dal_buffer.to_bytes();

    let parquet_builder =
        parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(dal_bytes)?;
    let mut parquet_reader = parquet_builder.build()?;

    let batch = parquet_reader
        .next()
        .ok_or_else(|| anyhow::anyhow!("No batches found"))??;

    // println!("{:?}", batch);

    let id_col = batch
        .column_by_name("id")
        .ok_or_else(|| anyhow::anyhow!("Column id missing"))?
        .as_primitive::<arrow::datatypes::Int32Type>();

    let input_data: &[i32] = id_col.values();
    println!("Result from File: {:?}", input_data);

    let row_count = batch.num_rows() as u32;
    let workgroup_count = (row_count + 63) / 64;
    let buffer_size = (row_count as usize * std::mem::size_of::<i32>()) as u64;

    // BUFFERS
    // let input_buffer = gpu.input_buffer("Input Buffer", &input_data);

    let output_buffer = gpu.output_buffer("Output Buffer", buffer_size);

    // GPU TO CPU BUFFER
    let stagging_buffer = gpu.stagging_buffer("Staging Buffer", buffer_size);

    // UNIFORM BUFFER
    let params = gpu::QueryParams { row_count };

    // Using storage buffer instead of uniform due to alignment errors
    let storage_buffer = gpu.metadata_buffer("Query Params", params);

    // JIT WGSL
    // ((id + 2) * 5) - 7
    let query = jit::Expression::Subtract(
        Box::new(jit::Expression::Multiply(
            Box::new(jit::Expression::Add(
                Box::new(jit::Expression::Column(0)),
                Box::new(jit::Expression::Literal(2)),
            )),
            Box::new(jit::Expression::Literal(5)),
        )),
        Box::new(jit::Expression::Literal(7)),
    );
    // Columns in the query
    let mut used_cols = std::collections::BTreeSet::new();
    jit::collect_columns(&query, &mut used_cols);

    // Map Columns to sequential bindings
    let mut mapping = std::collections::BTreeMap::new();
    for (binding_idx, col_idx) in used_cols.iter().enumerate() {
        mapping.insert(*col_idx, binding_idx as u32);
    }

    // Generate shadder from mapping
    let wgsl = jit::generate_shader(&query, &mapping);

    // LOAD SHADER
    let shader = gpu
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("JIT Projection Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&wgsl)),
        });

    // PIPELINE
    let compute_pipeline = gpu
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    // BIND GPOUP
    // input buffers for all columns used in query
    let mut input_buffers = Vec::new();
    for &col_idx in &used_cols {
        let col = batch
            .column(col_idx as usize)
            .as_primitive::<arrow::datatypes::Int32Type>();
        input_buffers.push(gpu.input_buffer(&format!("Col {}", col_idx), col.values()));
    }
    // Bind group entries from input buffers
    let mut entries = Vec::new();

    for (i, buffer) in input_buffers.iter().enumerate() {
        entries.push(wgpu::BindGroupEntry {
            binding: i as u32,
            resource: buffer.as_entire_binding(),
        });
    }

    // output buffer bindings at mapping.len
    let out_slot = mapping.len() as u32;
    entries.push(wgpu::BindGroupEntry {
        binding: out_slot,
        resource: output_buffer.as_entire_binding(),
    });

    // uniform buffer binding at mapping.len + 1
    let uniform_slot = out_slot + 1;
    entries.push(wgpu::BindGroupEntry {
        binding: uniform_slot,
        resource: storage_buffer.as_entire_binding(),
    });

    // create bind group
    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Dynamic Bind Group"),
        layout: &compute_pipeline.get_bind_group_layout(0),
        entries: &entries,
    });

    // EXECUTE
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
    }

    // COPY TO STAGGING BUFFER
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &stagging_buffer, 0, buffer_size);
    gpu.queue.submit(Some(encoder.finish()));

    // BACK 2 CPU
    let buffer_slice = stagging_buffer.slice(..);
    let (sender, receiver) = tokio::sync::oneshot::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
        sender
            .send(v)
            .expect("Unable to send Buffer back to the CPU")
    });

    let _ = gpu.device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });

    if let Ok(Ok(())) = receiver.await {
        let data = buffer_slice.get_mapped_range();
        let result: &[i32] = bytemuck::cast_slice(&data);
        println!("Result from GPU: {:?}", result);
        drop(data);
        stagging_buffer.unmap();
    }

    Ok(())
}
