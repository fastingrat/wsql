pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl Gpu {
    pub async fn new() -> Self {
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
            .await
            .expect("Unable to Request Device");

        Self { device, queue }
    }
}
