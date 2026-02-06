use std::collections::HashMap;

use prost::Message;
use substrait::proto::{
    Plan,
    expression::{RexType, reference_segment as direct_reference, field_reference},
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
        RexType::Literal(lit) => {
            if let Some(substrait::proto::expression::literal::LiteralType::I32(v)) =
                &lit.literal_type
            {
                Ok(jit::Expression::Literal(*v))
            } else {
                anyhow::bail!("Only I32 literals are supported")
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

            match func_name.as_str() {
                "add" => Ok(jit::Expression::Add(
                    Box::new(
                        args.next()
                            .ok_or_else(|| anyhow::anyhow!("Missing arg"))??,
                    ),
                    Box::new(
                        args.next()
                            .ok_or_else(|| anyhow::anyhow!("Missing arg"))??,
                    ),
                )),
                _ => anyhow::bail!("Unsupported function: {}", func_name),
            }
        }

        _ => anyhow::bail!("Unsupported Substrait type"),
    }
}
