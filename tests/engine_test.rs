#[tokio::test]
async fn test_engine_streaming_aggregate() {
    let gpu = wsql::gpu::Gpu::new().await;
    let executor = wsql::executor::QueryExecutor::new(gpu);
    let engine = wsql::engine::QueryEngine::new(executor);

    let dal_builder = opendal::services::Fs::default().root("tests");
    let op = opendal::Operator::new(dal_builder).unwrap().finish();

    let file_path = "data/alltypes_plain.parquet";
    let buffer = op.read(file_path).await.unwrap();
    let reader =
        parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(buffer.to_bytes())
            .unwrap()
            .with_batch_size(2) // forced streaming
            .build()
            .unwrap();

    let json_plan = std::fs::read_to_string("tests/fixtures/streaming_aggregate.json").unwrap();
    let result = engine.run(reader, &json_plan).await.unwrap();

    assert_eq!(result, wsql::executor::QueryResult::Aggregate(28.0));
}
