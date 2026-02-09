use crate::sub::PhysicalPlan;

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
    Date(i32),
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
            LiteralTypes::I32(v) | LiteralTypes::Date(v) => format!("{}i", v),
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

pub fn generate_shader(
    physical_plan: &PhysicalPlan,
    mapping: &std::collections::BTreeMap<u32, u32>,
) -> String {
    let logic = translate(&physical_plan.projection, mapping);

    // check for FILTER
    let condition = physical_plan
        .filter
        .as_ref()
        .map_or("true".into(), |f| translate(f, mapping));

    // data types
    let scalar_type = if physical_plan.is_aggregate {
        "f32"
    } else {
        "i32"
    };
    let sentinel = if physical_plan.is_aggregate {
        "0.0f"
    } else {
        // 'Dynamic Shader' parsing error: numeric literal not representable by target type: 2147483648i
        // Parser sees the +ive integer first and overflows
        // "-2147483648i"
        "bitcast<i32>(0x80000000u)"
    };

    let (globals, write_logic) = if physical_plan.is_aggregate {
        (
            "var<workgroup> scratch: array<f32, 64>;",
            r#"
                // move value into shared scratchpad
                scratch[l_idx] = val;

                // Sync: wait for all 64 threads to finish
                workgroupBarrier();

                // Reduction Tree
                // 32->16->...->1
                for (var s = 32u; s > 0u; s >>= 1u) {
                    if (l_idx < s) {
                        scratch[l_idx] += scratch[l_idx + s];
                    }

                    // Sync
                    workgroupBarrier();
                }
                // Write to the global memory
                if (l_idx == 0u) {
                    out_col[group_id.x] = scratch[0];
                }
            "#,
        )
    } else {
        ("", "if (idx < params.row_count) { out_col[idx] = val; }")
    };

    // shader bindings
    let mut bindings = String::new();

    // Input bindings for every column from the mapping
    for binding_idx in mapping.values() {
        bindings.push_str(&format!(
            "@group(0) @binding({binding_idx}) var<storage, read> in_col_{binding_idx}: array<{scalar_type}>;\n"
        ));
    }

    // output buffer
    let out_slot = mapping.len() as u32;
    bindings.push_str(&format!(
        "@group(0) @binding({out_slot}) var<storage, read_write> out_col: array<{scalar_type}>;\n",
    ));

    // uniform buffer
    bindings.push_str(&format!(
        "@group(0) @binding({uniform_slot}) var<storage, read> params: QueryParams;\n",
        uniform_slot = out_slot + 1
    ));

    // Final Assembly
    format!(
        r#"
        struct QueryParams {{
            row_count: u32,
        }}

        {bindings}
        {globals}

        @compute @workgroup_size(64)
        fn main(
            @builtin(global_invocation_id) global_id: vec3<u32>,
            @builtin(local_invocation_id) local_id: vec3<u32>,
            @builtin(workgroup_id) group_id: vec3<u32>
        ) {{
            let idx = global_id.x;
            let l_idx = local_id.x;

            // init with neutral element (0.0 for sum, i32::MIN for project)
            var val: {scalar_type} = {sentinel};

            // Calculate logic only for valid rows
            if (idx < params.row_count) {{
                if ({condition}) {{
                    val = {logic};
                }}
            }}

            {write_logic}
        }}
    "#,
    )
}
