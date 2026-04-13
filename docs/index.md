# SCIV Documentation

SCIV (Single-Cell Integrated Variational) is a comprehensive tool for single-cell omics data analysis, with a focus on integrating scATAC-seq data with genetic variant information.

## Table of Contents

- [Installation](sciv_usage.md#11-install)
- [Usage](sciv_usage.md#12-sciv-execution-process)
  - [Download scATAC seq sample data](sciv_usage.md#121-download-scatac-seq-sample-data)
  - [Download trait example data](sciv_usage.md#122-download-trait-example-data)
  - [Run SCIV](sciv_usage.md#123-run-sciv)
- [API Reference](sciv_usage.md#2-api-reference)
  - [Core Modules](sciv_usage.md#21-core-modules)
  - [API Workflow](sciv_usage.md#22-api-workflow)
  - [Key API Functions](sciv_usage.md#23-key-api-functions)

## Quick Start

```python
import sciv

# Load scATAC-seq data
adata = sciv.fl.read_h5ad(file="path/to/file.h5ad")

# Read variant data
variants, trait_info = sciv.fl.read_variants(
    variant_base_path="path/to/variants",
    column_map={0: "chr", 1: "position", 3: "rsId", 4: "pp"}
)

# Run SCIV core model
trs = sciv.ml.core(
    adata=adata,
    variants=variants,
    trait_info=trait_info,
    save_path="path/to/save/results",
    model_dir="path/to/save/model",
    is_file_exist_loading=True
)

print(trs)
```

## Features

- **Integration of scATAC-seq data with genetic variants**
- **Variational inference for dimensionality reduction**
- **Association score calculation**
- **Comprehensive visualization tools**
- **Efficient preprocessing utilities**

## License

SCIV is released under the MIT License. See the [LICENSE](https://github.com/yourusername/sciv/blob/main/LICENSE) file for details.

## Contributing

We welcome contributions from the community! Please see our [contributing guidelines](https://github.com/yourusername/sciv/blob/main/CONTRIBUTING.md) for more information.

## Contact

For questions, issues, or feature requests, please open an issue on our [GitHub repository](https://github.com/yourusername/sciv).