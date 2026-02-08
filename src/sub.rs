use std::collections::HashMap;

use prost::Message;
use substrait::proto::{
    Plan,
    expression::{RexType, field_reference, reference_segment as direct_reference},
    extensions::simple_extension_declaration::MappingType,
};

use crate::jit;

pub fn decode_plan(bytes: &[u8]) -> anyhow::Result<Plan> {
    Plan::decode(bytes).map_err(|e| anyhow::anyhow!("Failed to decode plan: {e}"))
}

pub fn get_functions_map(plan: &Plan) -> HashMap<u32, String> {
    let mut map = HashMap::new();
    for ext in &plan.extensions {
        if let Some(MappingType::ExtensionFunction(func)) = &ext.mapping_type {
            map.insert(func.function_anchor, func.name.clone());
        }
    }

    map
}

// Convert protobuf into JIT IR
pub fn lower_expression(
    substrait_expr: &substrait::proto::Expression,
    function_map: &HashMap<u32, String>,
) -> anyhow::Result<jit::Expression> {
    let rex = substrait_expr
        .rex_type
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Expression must have a type"))?;

    match rex {
        // Literals
        // I32
        // F32
        RexType::Literal(lit) => {
            let value = lit
                .literal_type
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("Literal missing value"))?;

            match value {
                substrait::proto::expression::literal::LiteralType::I32(v) => {
                    Ok(jit::Expression::Literal(jit::LiteralTypes::I32(*v)))
                }
                substrait::proto::expression::literal::LiteralType::Fp32(v) => {
                    Ok(jit::Expression::Literal(jit::LiteralTypes::F32(*v)))
                }
                _ => anyhow::bail!("Literal type {:?} is not supported yet", value),
            }
        }

        // Selection
        RexType::Selection(sel) => match sel.reference_type.as_ref() {
            Some(field_reference::ReferenceType::DirectReference(dr)) => {
                match dr.reference_type.as_ref() {
                    Some(direct_reference::ReferenceType::StructField(sf)) => {
                        Ok(jit::Expression::Column(sf.field as u32))
                    }
                    _ => anyhow::bail!("Expected StructField"),
                }
            }
            _ => anyhow::bail!("Expected DirectReference"),
        },

        // Scalar Functions
        // Add
        // Greater Than
        RexType::ScalarFunction(f) => {
            let func_name = function_map
                .get(&f.function_reference)
                .ok_or_else(|| anyhow::anyhow!("Unknown function: {}", f.function_reference))?;

            let mut args = f.arguments.iter().map(|a| {
                let arg_expr = a
                    .arg_type
                    .as_ref()
                    .and_then(|kind| match kind {
                        substrait::proto::function_argument::ArgType::Value(e) => Some(e),
                        _ => None,
                    })
                    .ok_or_else(|| anyhow::anyhow!("Function argument must be a value"))?;
                lower_expression(arg_expr, function_map)
            });

            let mut next_arg = || -> anyhow::Result<Box<jit::Expression>> {
                Ok(Box::new(args.next().ok_or_else(|| {
                    anyhow::anyhow!("Missing argyment for {}", func_name)
                })??))
            };

            match func_name.as_str() {
                "add" => Ok(jit::Expression::Add(next_arg()?, next_arg()?)),
                "sub" => Ok(jit::Expression::Subtract(next_arg()?, next_arg()?)),
                "mul" => Ok(jit::Expression::Multiply(next_arg()?, next_arg()?)),
                "gt" => Ok(jit::Expression::GreaterThan(next_arg()?, next_arg()?)),
                "lt" => Ok(jit::Expression::LessThan(next_arg()?, next_arg()?)),
                "and" => Ok(jit::Expression::And(next_arg()?, next_arg()?)),
                "or" => Ok(jit::Expression::Or(next_arg()?, next_arg()?)),
                _ => anyhow::bail!("Unsupported function: {}", func_name),
            }
        }

        _ => anyhow::bail!("Unsupported Substrait type"),
    }
}

pub fn get_project_expression(plan: &Plan) -> anyhow::Result<Vec<substrait::proto::Expression>> {
    // Get the root
    let rel = plan
        .relations
        .first()
        .ok_or_else(|| anyhow::anyhow!("Plan has no relations"))?
        .rel_type
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Relation has no type"))?;

    match rel {
        substrait::proto::plan_rel::RelType::Root(rel_root) => {
            let inner_rel = rel_root
                .input
                .as_ref()
                .and_then(|r| r.rel_type.as_ref())
                .ok_or_else(|| anyhow::anyhow!("Root has no input relation"))?;

            match inner_rel {
                substrait::proto::rel::RelType::Project(project_rel) => {
                    Ok(project_rel.expressions.clone())
                }
                _ => anyhow::bail!("Expected Project relation, found {:?}", inner_rel),
            }
        }
        _ => anyhow::bail!("Expected Root relation"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lower_add_expression() {
        let file = "tests/fixtures/simple_add.json";
        let json_plan = std::fs::read_to_string(file)
            .unwrap_or_else(|e| panic!("Unable to open {file} due to {e}"));
        let plan: substrait::proto::Plan =
            serde_json::from_str(&json_plan).expect("Something is wrong with json_plan");

        let fn_map = get_functions_map(&plan);
        let exprs = get_project_expression(&plan).expect("Couldnt get project expr");
        let jit_expr = lower_expression(&exprs[0], &fn_map).expect("Couldnt lower expr");

        match jit_expr {
            jit::Expression::Add(l, r) => {
                match *l {
                    jit::Expression::Column(idx) => assert_eq!(idx, 0),
                    _ => panic!("Let side of Add should be Coulmn 0"),
                }
                match *r {
                    jit::Expression::Literal(jit::LiteralTypes::I32(val)) => assert_eq!(val, 10),
                    _ => panic!("Rgiht side of Add should be 10"),
                }
            }

            _ => panic!("Expected Add expression"),
        }
    }

    #[test]
    fn test_lower_f32_and_logic() {
        let json_plan = std::fs::read_to_string("tests/fixtures/f32_logic.json").unwrap();
        let plan: substrait::proto::Plan = serde_json::from_str(&json_plan).unwrap();

        let fn_map = get_functions_map(&plan);
        let exprs = get_project_expression(&plan).unwrap();
        let jit_expr = lower_expression(&exprs[0], &fn_map).expect("Failed to lower f32 logic");

        match jit_expr {
            jit::Expression::And(l, r) => {
                match *l {
                    jit::Expression::GreaterThan(_, r) => {
                        if let jit::Expression::Literal(jit::LiteralTypes::F32(v)) = *r {
                            assert_eq!(v, 10.5);
                        } else {
                            panic!("Expected F32 literal 10.5");
                        }
                    }
                    _ => panic!("Expected GreaterTHan on left"),
                }
                match *r {
                    #[allow(clippy::assertions_on_constants)]
                    // Testing if substrate query is parsed correctly and outputs LessThan
                    jit::Expression::LessThan(_, _) => assert!(true),
                    _ => panic!("Expected LessThan on the right"),
                }
            }
            _ => panic!("Expected And expression"),
        }
    }
}
