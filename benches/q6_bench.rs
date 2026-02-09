use std::time::Instant;

use wsql::{engine::QueryEngine, executor::QueryExecutor, gpu::Gpu};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let gpu = Gpu::new().await;
    let executor = QueryExecutor::new(gpu);
    let engine = QueryEngine::new(executor);

    let dal_builder = opendal::services::Fs::default().root("benches");
    let op = opendal::Operator::new(dal_builder)
        .expect("Buffer")
        .finish();

    let file_path = "data/lineitem.parquet";
    let buffer = op.read(file_path).await.expect("Unable to create buffer");

    let reader =
        parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(buffer.to_bytes())
            .unwrap()
            .with_batch_size(65536)
            .build()
            .unwrap();

    let json_plan = std::fs::read_to_string("benches/queries/tpch_q6.json").unwrap();
    let start = Instant::now();
    let result = engine.run(reader, &json_plan).await?;
    let duration = start.elapsed();

    println!("Query time: {:?}", duration);
    println!("Query result: {:?}", result);

    Ok(())
}
