# SCIV Usage and API Reference

## 1. SCIV Usage

### 1.1 Install

```shell
conda create --name sciv python=3.10
conda activate sciv
pip install sciv
```

### 1.2 SCIV execution process

#### 1.2.1 Download scATAC seq sample data

Download PBMC case file：[GSM6793454_sc_atac_snapATAC2.h5ad](https://bio.liclab.net/scvmap_static/download/scatac/GSM6793454_sc_atac_snapATAC2.h5ad)

```shell
mkdir -p /project/sciv/input/scATAC/GSM6793454
cd /project/sciv/input/scATAC/GSM6793454
wget https://bio.liclab.net/scvmap_static/download/scatac/GSM6793454_sc_atac_snapATAC2.h5ad
```

#### 1.2.2 Download trait example data

Download the fine mapping results for monocytes, B cells, CD4+ and CD8+ T cells

```shell
mkdir -p /project/sciv/input/trait/GSM6793454
cd /project/sciv/input/trait/GSM6793454
wget https://bio.liclab.net/scvmap_static/sciv/download_example_traits.sh
chmod +x download_example_traits.sh
sh download_example_traits.sh
rm -rf download_example_traits.sh
```

#### 1.2.3 Run SCIV

Create Python file

```shell
mkdir -p /project/sciv/code/GSM6793454/
cd /project/sciv/code/GSM6793454/
touch sciv_pbmc.py
```

The file content is as follows

```python
# -*- coding: UTF-8 -*-

import os.path
import sciv

if __name__ == '__main__':

    # base path
    base_path: str = "/project/sciv"

    # path
    save_path: str = f"{base_path}/result/GSM6793454/data"

    # scATAC-seq data
    sc_atac_file = f"{base_path}/input/scATAC/GSM6793454/GSM6793454_sc_atac_snapATAC2.h5ad"
    sc_atac = sciv.fl.read_h5ad(file=sc_atac_file)

    # read variant information
    variant_base_path: str = f"{base_path}/input/trait/GSM6793454/hg19"
    variant_column_map: dict = {0: "chr", 1: "position", 3: "rsId", 4: "pp"}
    variants, trait_info = sciv.fl.read_variants(variant_base_path, column_map=variant_column_map)

    # run
    trs = sciv.ml.core(
        adata=sc_atac,
        variants=variants,
        trait_info=trait_info,
        save_path=save_path,
        model_dir=os.path.join(save_path, "poisson_vi"),
        is_file_exist_loading=True
    )

    print(trs)
```

Execute the file

```shell
python3 sciv_pbmc.py
```

The output log information is as follows

```shell
python3 sciv_pbmc.py
```

## 2. API Reference

### 2.1 Core Modules

SCIV provides several core modules for different functionalities:

- **sciv.fl** - File I/O operations
- **sciv.ml** - Machine learning models
- **sciv.pl** - Data visualization
- **sciv.pp** - Data preprocessing
- **sciv.tl** - Tools and algorithms
- **sciv.ul** - Utility functions

### 2.2 API Workflow

#### 2.2.1 Data Loading

```python
# Load scATAC-seq data
import sciv

# Read h5ad file
adata = sciv.fl.read_h5ad(file="path/to/file.h5ad")

# Read variant data
variants, trait_info = sciv.fl.read_variants(
    variant_base_path="path/to/variants",
    column_map={0: "chr", 1: "position", 3: "rsId", 4: "pp"}
)
```

#### 2.2.2 Model Execution

```python
# Run SCIV core model
trs = sciv.ml.core(
    adata=adata,
    variants=variants,
    trait_info=trait_info,
    save_path="path/to/save/results",
    model_dir="path/to/save/model",
    is_file_exist_loading=True
)
```

#### 2.2.3 Data Preprocessing

```python
# Normalize data
normalized_data = sciv.pp.normalize_data(adata)

# Scale data
scaled_data = sciv.pp.scale_data(normalized_data)

# Filter genes
filtered_data = sciv.pp.filter_genes(adata, min_counts=10)
```

#### 2.2.4 Visualization

```python
# UMAP plot
sciv.pl.umap_plot(adata, color="cell_type")

# Heatmap
sciv.pl.heatmap(adata, genes=["gene1", "gene2", "gene3"])

# Volcano plot
sciv.pl.volcano_plot(results, pvalue_threshold=0.05)
```

#### 2.2.5 Utility Functions

```python
# Convert to dense matrix
dense_matrix = sciv.ul.to_dense(sparse_matrix)

# Convert to sparse matrix
sparse_matrix = sciv.ul.to_sparse(dense_matrix)

# Get memory usage
def my_function():
    # function code
    pass

result = sciv.ul.track_with_memory()(my_function)()
print(f"Execution time: {result['time']} seconds")
print(f"Memory usage: {max(result['memory']) / 1e6} MB")
```

### 2.3 Key API Functions

#### File Module (sciv.fl)

- **read_h5ad(file)**: Read h5ad file
- **read_h5(file)**: Read h5 file
- **read_pkl(file)**: Read pickle file
- **read_csv(file)**: Read CSV file
- **read_bed(file)**: Read BED file
- **read_tsv(file)**: Read TSV file
- **read_gtf(file)**: Read GTF file
- **read_fasta(file)**: Read FASTA file
- **save_h5ad(adata, file)**: Save AnnData to h5ad file
- **save_h5(data, file)**: Save data to h5 file
- **save_pkl(data, file)**: Save data to pickle file
- **to_meta(adata)**: Convert AnnData to metadata DataFrame
- **to_fragments(adata)**: Convert AnnData to fragments format

#### Model Module (sciv.ml)

- **core(adata, variants, trait_info, save_path, model_dir, is_file_exist_loading)**: Core SCIV model
- **association_score(adata, variants)**: Calculate association scores
- **knock(adata, variants, target_genes)**: Perform knock analysis

#### Preprocessing Module (sciv.pp)

- **poisson_vi(adata)**: Poisson variational inference
- **gsea_enrichr(genes, database)**: GSEA enrichment analysis
- **adata_map_df(adata)**: Convert AnnData to map DataFrame
- **adata_group(adata, groupby)**: Group AnnData by specified column
- **normalize_data(adata)**: Normalize data
- **scale_data(adata)**: Scale data
- **filter_genes(adata, min_counts)**: Filter genes
- **filter_cells(adata, min_genes)**: Filter cells
- **select_features(adata, n_features)**: Select features
- **batch_correction(adata, batch_key)**: Batch correction
- **impute_data(adata)**: Impute missing data
- **feature_selection(adata, method)**: Feature selection
- **dimensionality_reduction(adata, method, n_components)**: Dimensionality reduction
- **cell_cycle_scoring(adata)**: Cell cycle scoring

#### Tool Module (sciv.tl)

- **sigmoid(data)**: Sigmoid function
- **tf_idf(data)**: TF-IDF transformation
- **z_score_normalize(data)**: Z-score normalization
- **lsi(data, n_components)**: Latent Semantic Indexing
- **pca(data, n_components)**: Principal Component Analysis
- **tsne(data, n_components)**: t-SNE dimensionality reduction
- **umap(data, n_components)**: UMAP dimensionality reduction
- **k_means(data, n_clusters)**: K-means clustering
- **spectral_clustering(data, n_clusters)**: Spectral clustering
- **overlap(regions, variants)**: Calculate overlap between regions and variants
- **euclidean_distances(data1, data2)**: Calculate Euclidean distances

#### Plot Module (sciv.pl)

- **group_heatmap(adata, groupby)**: Group heatmap
- **map_df_plot(map_df)**: Map DataFrame plot
- **volcano_plot(results, pvalue_threshold)**: Volcano plot
- **umap_plot(adata, color)**: UMAP plot
- **tsne_plot(adata, color)**: t-SNE plot
- **pca_plot(adata, color)**: PCA plot
- **scatter_plot(x, y, color)**: Scatter plot
- **bar_plot(data, x, y)**: Bar plot
- **box_plot(data, x, y)**: Box plot
- **violin_plot(data, x, y)**: Violin plot
- **heatmap(data)**: Heatmap
- **network_plot(adj_matrix, nodes)**: Network plot

#### Util Module (sciv.ul)

- **file_method(name)**: File method utility
- **log(name)**: Logging utility
- **track_with_memory(interval)**: Memory tracking decorator
- **to_dense(sm)**: Convert to dense matrix
- **to_sparse(dm)**: Convert to sparse matrix
- **sum_min_max(data)**: Calculate sum min/max
- **get_index(position, positions_list)**: Get index using binary search
- **list_duplicate_set(data)**: Handle duplicate values in list
- **split_matrix(matrix, n_splits)**: Split matrix
- **merge_matrix(matrices)**: Merge matrices
- **check_gpu_availability()**: Check GPU availability