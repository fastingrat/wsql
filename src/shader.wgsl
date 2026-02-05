@group(0) @binding(0) var<storage, read> input_array: array<i32>;
@group(0) @binding(1) var<storage, read_write> output_array: array<i32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    output_array[idx] = input_array[idx] + 10;
}