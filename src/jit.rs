pub enum Expression {
    Literal(i32),
    Column(u32),
    Add(Box<Expression>, Box<Expression>),
    Subtract(Box<Expression>, Box<Expression>),
    Multiply(Box<Expression>, Box<Expression>),
}

pub fn translate(expr: &Expression) -> String {
    match expr {
        Expression::Literal(v) => format!("{}i", v),
        Expression::Column(i) => format!("in_col_{}[idx]", i),
        Expression::Add(l, r) => format!("({} + {})", translate(l), translate(r)),
        Expression::Subtract(l, r) => format!("({} - {})", translate(l), translate(r)),
        Expression::Multiply(l, r) => format!("({} * {})", translate(l), translate(r)),
    }
}

pub fn generate_shader(expr: &Expression, col_count: u32, row_count: u32) -> String {
    let mut bindings = String::new();

    // Input bindings for every column being used
    for i in 0..col_count {
        bindings.push_str(&format!(
            "@group(0) @binding({}) var<storage, read> in_col_{}: array<i32>;\n",
            i, i
        ));
    }

    // output buffer
    bindings.push_str(&format!(
        "@group(0) @binding({}) var<storage, read_write> out_col: array<i32>;\n",
        col_count
    ));

    let logic = translate(expr);

    format!(
        r#"
            {bindings}

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
                let idx = global_id.x;
                if (idx >= {row_count}u) {{ return; }}
                out_col[idx] = {logic};
            }}
        "#
    )
}
