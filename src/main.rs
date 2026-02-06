use arrow::array::AsArray;
use wsql;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // GPU
    let gpu = wsql::gpu::Gpu::new().await;
    // DATA
    let mut dal_builder = opendal::services::Fs::default().root(".");
    let dal_op = opendal::Operator::new(dal_builder)?.finish();
    let dal_buffer = dal_op.read("data/alltypes_plain.parquet").await?;
    let dal_bytes = dal_buffer.to_bytes();

    let parquet_builder =
        parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(dal_bytes)?;
    let mut parquet_reader = parquet_builder.build()?;

    let batch = parquet_reader
        .next()
        .ok_or_else(|| anyhow::anyhow!("No batches found"))??;

    // println!("{:?}", batch);

    let id_col = batch
        .column_by_name("id")
        .ok_or_else(|| anyhow::anyhow!("Column id missing"))?
        .as_primitive::<arrow::datatypes::Int32Type>();

    let input_data: &[i32] = id_col.values();
    println!("Result from File: {:?}", input_data);

    // JIT WGSL
    // ((id + 2) * 5) - 7
    let query = wsql::jit::Expression::Subtract(
        Box::new(wsql::jit::Expression::Multiply(
            Box::new(wsql::jit::Expression::Add(
                Box::new(wsql::jit::Expression::Column(0)),
                Box::new(wsql::jit::Expression::Literal(2)),
            )),
            Box::new(wsql::jit::Expression::Literal(5)),
        )),
        Box::new(wsql::jit::Expression::Literal(7)),
    );

    let executor = wsql::executor::QueryExecutor::new(gpu);
    let result = executor.execute_batch(&batch, &query).await?;

    println!("Result from GPU: {:?}", result);

    Ok(())
}
