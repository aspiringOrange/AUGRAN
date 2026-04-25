# AUGRAN

A **Dynamic Range-Filtering Approximate Nearest Neighbor (RFANN) Index** based on a **hierarchical segmented subgraph architecture**.
It is specifically designed for **range-filtering scenarios**, supporting efficient **vector insertion** and **attribute updates**.

---

## Requirements

* A modern C++ compiler with **C++11 support**
* **CMake ≥ 3.20**

---

## Build

```bash
mkdir build
cd build
cmake ..
make -j
```

---

## Usage

### 1. Index Construction

```bash
./build --m 16 --efc 256 --rebuild_ratio 1.5 \
  --basevec /dataset/vector.fvecs \
  --baseatt /dataset/attribute.bin \
  --index_location /data/index
```

#### Parameters

* `--m`
  Maximum number of neighbors per node in the graph.
  Controls memory usage and affects query performance.
  **Recommended: 16**

* `--efc`
  Candidate list size during graph construction.
  Larger values improve graph quality and query accuracy but increase build time.
  **Recommended: 256**

* `--rebuild_ratio`
  Threshold for subgraph rebuilding in attribute update scenarios (must be > 1).
  Smaller values lead to:

  * more frequent rebuilds
  * lower memory overhead
  * better query performance
  * but higher update latency
  
    **Recommended: 1.5**

* `--basevec`
  Base vector dataset in **fvecs format**:

  ```
  (num_vectors, dim, [], dim, [], ...)
  ```

  See `fvecs_read` in `bench_utils.hh`.

* `--baseatt`
  Attribute values corresponding to each vector (row-aligned):

  ```
  (att1, att2, att3, ...)
  ```

  See `LoadAttVec` in `bench_utils.hh`.

* `--index_location`
  Path to store the built index.

---

### 2. Attribute Update

```bash
./update \
  --index_location /data/index \
  --update_file /data/update.bin
```

#### Parameters

* `--index_location`
  Path to the existing index.

* `--update_file`
  Incremental attribute updates in the format:

  ```
  (id1, new_att1, id2, new_att2, ...)
  ```

  where `id ∈ [0, n)` corresponds to the row index in the base dataset.
  See `load_increment_update` in `bench_utils.hh`.

---

### 3. Range-Filtering ANN Search

```bash
./query --k 10 \
  --query_vec /data/query.fvecs \
  --query_rng /data/range.bin \
  --gt_file /data/groundtruth.bin \
  --index_location /data/index
```

#### Parameters

* `--k`
  Return **top-k nearest neighbors**.

* `--query_vec`
  Query vectors in **fvecs format**.
  See `fvecs_read` in `bench_utils.hh`.

* `--query_rng`
  Range filters aligned with queries:

  ```
  (l1, r1, l2, r2, ...)
  ```

  Each is a **closed interval [l, r]**.
  See `LoadRange` in `bench_utils.hh`.

* `--gt_file`
  Ground truth nearest neighbors:

  ```
  (k, id1, id2, ..., idk, k, id1, ...)
  ```

  Each group is sorted by distance.
  See `LoadGroundTruth` in `bench_utils.hh`.

* `--index_location`
  Path to the index.

---

## Datasets

| Dataset | Type  | Dim | Attribute      |
| ------- | ----- | --- | -------------- |
| ArXiv   | Text  | 384 | Update date    |
| YTaudio | Audio | 128 | Like count     |
| Sift    | Image | 128 | Random integer |
| Deep    | Image | 96  | Vector ID      |

### Links

* ArXiv: [https://github.com/qdrant/ann-filtering-benchmark-datasets](https://github.com/qdrant/ann-filtering-benchmark-datasets)
* YTaudio: [https://research.google.com/youtube8m/download.html](https://research.google.com/youtube8m/download.html)
* Sift: [http://corpus-texmex.irisa.fr](http://corpus-texmex.irisa.fr)
* Deep: [https://research.yandex.com/blog/benchmarks-for-billion-scale-similarity-search](https://research.yandex.com/blog/benchmarks-for-billion-scale-similarity-search)

---

## Notes

* The system is optimized for **range filtering scenarios** where attribute values may change over time.
* Combines **graph-based ANN search** with **attribute-aware filtering**.
* Designed for **high attribute update throughput** and **efficient query latency**.

---
