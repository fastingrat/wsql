Inspired by [Sirius](https://vldb.org/cidrdb/papers/2026/p12-yogatama.pdf), wsql is a SQL engine that leverages WebGPU instead of the NVIDIA ecosystem.

## Architecture
<!-- ![Architecture](.github/images/wSQL%201.png) -->
<p align="center">
  <img src=".github/images/wSQL 1.png" alt="Diagram" width="900"/>
</p>

## Benchmarking
1. Generate data - `duckdb -c "INSTALL tpch; LOAD tpch; CALL dbgen(sf=1); COPY lineitem TO 'benches/data/lineitem.parquet' (FORMAT PARQUET);"`
2. Run Bench - `cargo bench --bench q6_bench`

### Results

|Run   | Query time | Query result  |
|------|------------|---------------|
|Target| <200ms     | 1793214130.04 |
|Base  | ~580ms     | 1793214100.0  |
|LHF 1 | ~560ms     | 1793214100.0  |

### Low Hanging Fruits
1. [x] Either move Decimal128 downcast to GPU or add support for it.
    - Even with wgpu extension natively Decimal128 isnt supported, downcasting to f64 is possible and more precise but it will be both complex and slow.
    - Downcasting to f32
2. [ ] Increase batch size from 64 or make it dynamic.
3. [ ] Use same buffer across batches
4. [ ] Prep n+1 batch while n is executing on GPU
5. [ ] Use Better way to poll GPU
6. [ ] Remove the hacky bits

<details>
    <summary> TPCH Q6 bench progress (Completed)</summary>

### TPC-H Q6
#### Step 1: Basics
- [x] **OpenDAL / Parquet Integration:** Streaming bytes to CPU.
- [x] **Arrow Memory Layout:** Standardized buffer management.
- [x] **Headless WGPU Setup:** GPU compute without a window.
- [x] **JIT Compiler:** WGSL generation for arithmetic.
- [x] **Dynamic Bind Groups:** Automated mapping of columns to bindings.
- [x] **Metadata/Uniforms:** Passing `row_count` to GPU.

#### Step 2: Scalar & Predicate Logic
- [x] **Comparison Operators:** `>` and `<` in JIT.
- [x] **Fused Filtering:** Early exit in shaders (Selection Vectors).
- [x] **Substrait Expression Lowering:** Mapping Protobuf to IR.
- [x] **Boolean Logic:** `AND` / `OR` in JIT (Needed for Q6's 4 concurrent filters).
- [x] **Data Types:** Support for `f32` or `Decimal` (Q6 uses prices/discounts).
- [x] **Date Handling:** Treating dates as `i32` integers for comparison.

#### Step 3: Aggregations
- [x] **The "Sum" Kernel:** Using `atomicAdd` in WGSL to sum the results of the project.
    - Use **Shared Memory Tree** instead of hacking f32 into i32 for `atomicAdd`
- [x] **Partial Reductions:** Reducing 6 million rows to 1 row efficiently (Parallel Reduction).
- [x] **Global Result Buffer:** Downloading a single number instead of a whole column.

#### Step 4: Substrait Relational
- [x] **Relational Walker:** Handling `ReadRel` -> `FilterRel` -> `ProjectRel` -> `AggregateRel`.
- [x] **Schema Mapping:** Linking Parquet column names (e.g., `l_quantity`) to Substrait indices.

#### Step 5: Benchmarking & Scale
- [x] **TPC-H Data Gen:** Generating `lineitem.parquet` at Scale Factor 1 (6 million rows).
- [x] **Benchmarking Harness:** Measuring end-to-end time (I/O + Upload + Compute + Download).
- [x] **Comparative Analysis:** Comparing against **DuckDB** (CPU) and **DataFusion**.

</details>