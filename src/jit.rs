pub enum Expression {
    Literal(LiteralTypes),
    Column(u32),
    Add(Box<Expression>, Box<Expression>),
    Subtract(Box<Expression>, Box<Expression>),
    Multiply(Box<Expression>, Box<Expression>),
    GreaterThan(Box<Expression>, Box<Expression>),
    LessThan(Box<Expression>, Box<Expression>),
    Equal(Box<Expression>, Box<Expression>),
    And(Box<Expression>, Box<Expression>),
    Or(Box<Expression>, Box<Expression>),
}

pub enum LiteralTypes {
    I32(i32),
    F32(f32),
}

pub fn collect_columns(expr: &Expression, cols: &mut std::collections::BTreeSet<u32>) {
    match expr {
        Expression::Column(i) => {
            cols.insert(*i);
        }
        Expression::Add(l, r)
        | Expression::Subtract(l, r)
        | Expression::Multiply(l, r)
        | Expression::GreaterThan(l, r)
        | Expression::LessThan(l, r)
        | Expression::Equal(l, r)
        | Expression::And(l, r)
        | Expression::Or(l, r) => {
            collect_columns(l, cols);
            collect_columns(r, cols);
        }
        Expression::Literal(_) => {}
    }
}

pub fn translate(expr: &Expression, mapping: &std::collections::BTreeMap<u32, u32>) -> String {
    match expr {
        Expression::Literal(val) => match val {
            LiteralTypes::I32(v) => format!("{}i", v),
            LiteralTypes::F32(v) => format!("{}f", v),
        },
        Expression::Column(i) => {
            let binding_idx = mapping.get(i).expect("Column mapping missing");
            format!("in_col_{}[idx]", binding_idx)
        }
        Expression::Add(l, r) => format!("({} + {})", translate(l, mapping), translate(r, mapping)),
        Expression::Subtract(l, r) => {
            format!("({} - {})", translate(l, mapping), translate(r, mapping))
        }
        Expression::Multiply(l, r) => {
            format!("({} * {})", translate(l, mapping), translate(r, mapping))
        }
        Expression::GreaterThan(l, r) => {
            format!("({} > {})", translate(l, mapping), translate(r, mapping))
        }
        Expression::LessThan(l, r) => {
            format!("({} < {})", translate(l, mapping), translate(r, mapping))
        }
        Expression::Equal(l, r) => {
            format!("({} == {})", translate(l, mapping), translate(r, mapping))
        }
        Expression::And(l, r) => {
            format!("({} && {})", translate(l, mapping), translate(r, mapping))
        }
        Expression::Or(l, r) => {
            format!("({} || {})", translate(l, mapping), translate(r, mapping))
        }
    }
}

pub fn generate_fused_shader(
    projection: &Expression,
    filter: Option<&Expression>,
    mapping: &std::collections::BTreeMap<u32, u32>,
) -> String {
    let logic = translate(projection, mapping);

    // check for FILTER
    let condition = match filter {
        Some(expr) => translate(expr, mapping),
        None => "true".to_string(),
    };
    let mut bindings = String::new();

    // Input bindings for every column from the mapping
    for binding_idx in mapping.values() {
        bindings.push_str(&format!(
            "@group(0) @binding({}) var<storage, read> in_col_{}: array<i32>;\n",
            binding_idx, binding_idx
        ));
    }

    // output buffer
    let out_slot = mapping.len() as u32;
    bindings.push_str(&format!(
        "@group(0) @binding({out_slot}) var<storage, read_write> out_col: array<i32>;\n",
    ));

    // uniform buffer
    bindings.push_str(&format!(
        "@group(0) @binding({uniform_slot}) var<storage, read> params: QueryParams;\n",
        uniform_slot = out_slot + 1
    ));

    format!(
        r#"
        struct QueryParams {{
            row_count: u32,
        }}

        {bindings}

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
            let idx = global_id.x;
            if (idx >= params.row_count) {{ return; }}

            // Fused shader: Early exit if filter fails
            if ({condition}) {{
                out_col[idx] = {logic};
            }} else {{
                out_col[idx] = -2147483648; // Filtered out i32::Min
            }}
        }}
    "#
    )
}

pub fn generate_shader(
    expr: &Expression,
    mapping: &std::collections::BTreeMap<u32, u32>,
) -> String {
    let mut bindings = String::new();

    // Input bindings for every column from the mapping
    for binding_idx in mapping.values() {
        bindings.push_str(&format!(
            "@group(0) @binding({}) var<storage, read> in_col_{}: array<i32>;\n",
            binding_idx, binding_idx
        ));
    }

    // output buffer
    let out_slot = mapping.len() as u32;
    bindings.push_str(&format!(
        "@group(0) @binding({out_slot}) var<storage, read_write> out_col: array<i32>;\n",
    ));

    // uniform buffer
    bindings.push_str(&format!(
        "@group(0) @binding({uniform_slot}) var<storage, read> params: QueryParams;\n",
        uniform_slot = out_slot + 1
    ));

    let logic = translate(expr, mapping);

    format!(
        r#"
            struct QueryParams {{
                row_count: u32,
            }}

            {bindings}

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
                let idx = global_id.x;
                if (idx >= params.row_count) {{ return; }}
                out_col[idx] = {logic};
            }}
        "#
    )
}
