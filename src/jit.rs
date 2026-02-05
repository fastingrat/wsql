pub enum Operation {
    Add(i32),
    Subtract(i32),
    Multiply(i32),
}

pub struct ShaderBuilder;

impl ShaderBuilder {
    pub fn build_projection(op: Operation) -> String {
        let op_str = match op {
            Operation::Add(val) => format!("+ {}", val),
            Operation::Subtract(val) => format!("- {}", val),
            Operation::Multiply(val) => format!("* {}", val),
        };

        format!(
            r#"
            @group(0) @binding(0) var<storage, read>       input_array:  array<i32>;
            @group(0) @binding(1) var<storage, read_write> output_array: array<i32>;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
                let idx = global_id.x;
                // Boundary check
                if (idx >= arrayLength(&output_array)) {{
                    return;
                }}

                output_array[idx] = input_array[idx] {};
            }}
        "#,
            op_str
        )
    }
}
