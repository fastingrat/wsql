use std::borrow::Cow;
use wgpu::util::DeviceExt;

async fn run_query() -> anyhow::Result<()> {
    // GPU
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .expect("Failed to get an adapter");

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await?;

    // DATA
    let input_data: Vec<i32> = vec![20, 25, 30, 35, 40];
    let size = (input_data.len() * std::mem::size_of::<i32>()) as u64;

    // BUFFERS
    let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Input Buffer"),
        contents: bytemuck::cast_slice(&input_data),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer"),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // GPU TO CPU BUFFER
    let stagging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // LOAD SHADER
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Projection Shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
    });

    // PIPELINE
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute Pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // BIND GPOUP
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
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
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

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
    queue.submit(Some(encoder.finish()));

    // BACK 2 CPU
    let buffer_slice = stagging_buffer.slice(..);
    let (sender, receiver) = tokio::sync::oneshot::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    let _ = device.poll(wgpu::PollType::Wait {
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

fn main() {
    pollster::block_on(run_query()).unwrap();
}
