Inspired by [Sirius](https://vldb.org/cidrdb/papers/2026/p12-yogatama.pdf), wsql is a SQL engine that leverages WebGPU instead of the NVIDIA ecosystem.

## Architecture
<!-- ![Architecture](.github/images/wSQL%201.png) -->
<p align="center">
  <img src=".github/images/wSQL 1.png" alt="Diagram" width="900"/>
</p>

### Current Target - TPC-H Q6
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
- [ ] **Relational Walker:** Handling `ReadRel` -> `FilterRel` -> `ProjectRel` -> `AggregateRel`.
- [ ] **Schema Mapping:** Linking Parquet column names (e.g., `l_quantity`) to Substrait indices.

#### Step 5: Benchmarking & Scale
- [ ] **TPC-H Data Gen:** Generating `lineitem.parquet` at Scale Factor 1 (6 million rows).
- [ ] **Benchmarking Harness:** Measuring end-to-end time (I/O + Upload + Compute + Download).
- [ ] **Comparative Analysis:** Comparing against **DuckDB** (CPU) and **DataFusion**.

