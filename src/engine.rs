use arrow::array::RecordBatchReader;
use parquet::arrow::arrow_reader::ParquetRecordBatchReader;

use crate::{executor, sub};

pub struct QueryEngine {
    executor: executor::QueryExecutor,
}

impl QueryEngine {
    pub fn new(executor: executor::QueryExecutor) -> Self {
        Self { executor }
    }

    pub async fn run(
        &self,
        reader: ParquetRecordBatchReader,
        json_plan: &str,
    ) -> anyhow::Result<executor::QueryResult> {
        let plan: substrait::proto::Plan = serde_json::from_str(json_plan)?;
        let mut physical_plan = sub::lower_plan(&plan)?;

        let schema = reader.schema();
        for (i, field) in schema.fields().iter().enumerate() {
            let arrow_type = match field.data_type() {
                arrow::datatypes::DataType::Decimal128(_, _) => arrow::datatypes::DataType::Float32,
                other => other.clone(),
            };

            physical_plan.column_types.insert(i as u32, arrow_type);
        }
        let compiled = self.executor.compile(physical_plan)?;

        let mut global_results: Option<executor::QueryResult> = None;

        // stream batches
        for batch_res in reader {
            let batch = batch_res?;
            let batch_out = self.executor.execute(&compiled, &batch).await?;

            if let Some(ref mut global) = global_results {
                global.accumulate(batch_out)?;
            } else {
                global_results = Some(batch_out);
            }
        }

        global_results.ok_or_else(|| anyhow::anyhow!("No data processed"))
    }
}
