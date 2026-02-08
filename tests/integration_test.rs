use arrow::array::AsArray;
#[tokio::test]
async fn test_simple_substrait_lite() {
    // GPU
    let gpu = wsql::gpu::Gpu::new().await;
    let executor = wsql::executor::QueryExecutor::new(gpu);

    // DATA
    #[allow(unused_mut)] // https://docs.rs/opendal/0.55.0/opendal/#init-a-service
    let mut dal_builder = opendal::services::Fs::default().root("tests");
    let dal_op = opendal::Operator::new(dal_builder)
        .expect("Unable to create new OpenDAL Operator")
        .finish();
    let dal_buffer = dal_op
        .read("data/alltypes_plain.parquet")
        .await
        .expect("Unable to load file");
    let dal_bytes = dal_buffer.to_bytes();

    let parquet_builder =
        parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(dal_bytes)
            .expect("Couldnt build parquet from file");
    let mut parquet_reader = parquet_builder.build().expect("Couldnt read file");

    let batch = parquet_reader
        .next()
        .ok_or_else(|| anyhow::anyhow!("No batches found"))
        .unwrap()
        .unwrap();

    // println!("{:?}", batch);

    let id_col = batch
        .column_by_name("id")
        .ok_or_else(|| anyhow::anyhow!("Column id missing"))
        .unwrap()
        .as_primitive::<arrow::datatypes::Int32Type>();

    let input_data: &[i32] = id_col.values();
    // println!("Result from File: {:?}", input_data);
    assert_eq!(input_data, &[4, 5, 6, 7, 2, 3, 0, 1]);

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

    let result = executor.execute_batch(&batch, &query, None).await.unwrap();

    assert_eq!(result, vec![23, 28, 33, 38, 13, 18, 3, 8])

    // println!("Result from GPU: {:?}", result);
}

#[tokio::test]
async fn test_simple_filter_sparse() {
    use std::sync::Arc;
    use wsql::jit::Expression;

    let gpu = wsql::gpu::Gpu::new().await;
    let executor = wsql::executor::QueryExecutor::new(gpu);

    // DATA
    let schema = std::sync::Arc::new(arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("id", arrow::datatypes::DataType::Int32, false),
    ]));

    let batch = arrow::array::RecordBatch::try_new(
        schema,
        vec![Arc::new(arrow::array::Int32Array::from(vec![
            8, 9, 10, 11, 12, 13, 14, 15, 16,
        ]))],
    )
    .unwrap();

    // QUERY: select id where id > 12
    let projection = Expression::Column(0);
    let query = Expression::GreaterThan(
        Box::new(Expression::Column(0)),
        Box::new(Expression::Literal(12)),
    );

    // [-2147483648, -2147483648, -2147483648, -2147483648, -2147483648, 13, 14, 15, 16]
    let result = executor
        .execute_batch(&batch, &projection, Some(&query))
        .await
        .unwrap();

    assert_eq!(result[0], -2147483648);
    assert_eq!(result[4], -2147483648);
    assert_eq!(result[5], 13);
    assert_eq!(result[6], 14);
    assert_eq!(result[7], 15);
    assert_eq!(result[8], 16);
}
