use arrow::array::AsArray;
use std::borrow::Cow;

mod gpu;
mod jit;

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
        .expect("Column id missing")
        .as_primitive::<arrow::datatypes::Int32Type>();

    let input_data: &[i32] = id_col.values();
    println!("Result from File: {:?}", input_data);
    let size = (input_data.len() * std::mem::size_of::<i32>()) as u64;

    // BUFFERS
    let input_buffer = gpu.input_buffer("Input Buffer", &input_data);

    let output_buffer = gpu.output_buffer("Output Buffer", size);

    // GPU TO CPU BUFFER
    let stagging_buffer = gpu.stagging_buffer("Staging Buffer", size);

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
    let wgsl = jit::generate_shader(&query, 1);

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
    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &compute_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
        ],
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
        compute_pass.dispatch_workgroups(input_data.len() as u32, 1, 1);
    }

    // COPY TO STAGGING BUFFER
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &stagging_buffer, 0, size);
    gpu.queue.submit(Some(encoder.finish()));

    // BACK 2 CPU
    let buffer_slice = stagging_buffer.slice(..);
    let (sender, receiver) = tokio::sync::oneshot::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

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
