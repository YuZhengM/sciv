1.	SCIV usage
=========================

1.1 Standard pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^

1.1.1 Import library and environment setup
********************************************

Create environment and download SCIV package.

.. code-block:: shell

    conda create --name sciv python=3.12
    conda activate sciv
    pip install sciv


Import package and view version information.

.. code-block:: python

    import sciv

    sciv.__version__


1.1.2 Download example files
***************************************

We need to download the scATAC-seq and fine-mapping result files. These two files can be implemented by calling the following functions.

Download PBMC case file： `GSE139369_ELM_sim_snapATAC2.h5ad <https://bio.liclab.net/scvmap_static/sciv/GSE139369_ELM_sim_snapATAC2.h5ad>`_

.. code-block:: python

    adata = sciv.dl.read_sc_atac_file()

Download the fine-mapping results for monocytes, red blood cells, CD4+ and CD8+ T cells.

.. code-block:: python

    variants, trait_info = sciv.dl.read_trait_file()

1.1.3 Run SCIV
*****************

Obtain TRS results by executing the SCIV process using the sciv.ml.core function.

(1) Create Python file:

.. code-block:: shell

    touch sciv_pbmc.py

(2) The file content is as follows:

.. code-block:: python

    # -*- coding: UTF-8 -*-

    import sciv

    if __name__ == '__main__':

        # scATAC-seq data
        adata = sciv.dl.read_sc_atac_file()

        # read variant information
        variants, trait_info = sciv.dl.read_trait_file()

        # run
        trs = sciv.ml.core(
            adata=adata,
            variants=variants,
            trait_info=trait_info,
            cell_rate=0.05,    # This parameter speeds up execution for testing purposes. Remove or adjust it appropriately in production use cases.
            peak_rate=0.001,   # This parameter speeds up execution for testing purposes. Remove or adjust it appropriately in production use cases.
            save_path="./result",
            model_dir="./result/poisson_vi",
            is_file_exist_loading=True
        )

        print(trs)

(3) Executable the file:

.. code-block:: shell

    python3 sciv_pbmc.py

View TRS result:

.. code-block:: shell

    AnnData object with n_obs × n_vars = 1173 × 91
    obs: 'sample', 'barcodes', 'barcode', 'n_genes', 'n_counts', '_scvi_batch', '_scvi_labels', 'clusters', 'latent_umap1', 'latent_umap2', 'latent_tsne1', 'latent_tsne2'
    var: 'id', 'pp_sum', 'pp_mean', 'count', 'filename', 'labels', 'seed_cell_count', 'seed_cell_threshold'
    uns: 'is_sample', 'elapsed_time', 'cluster_info', 'params'
    layers: 'init_trs', 'seed_cell_index', 'trs_source', 'tre'

1.2 SCIV execution process for each step
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Import SCIV package to view version and set cache file path.

.. code-block:: python
    import sciv

    sciv.__version__

    sciv.ul.project_cache_path


1.2.1 Read a file
*****************

Read the scATAC-seq file using the ``sciv.fl.read_sc_atac`` function. This function supports reading resources in the following formats:
    1. Path to directory containing matrix, bed file, etc.
    2. H5 file obtained through cell-ranger
    3. A comprehensive h5ad file
    4. A table file with cell or peak columns and indexes, where content is fragment counts

When reading an H5AD file, ensure that the ``.var_name`` attribute of the file contains peak information (e.g. chr1:21234-34123), where the delimiter can be specified using the ``peak_split_character`` parameter.

.. code-block:: python
    import sciv
    import os

    sciv.dl.download_sc_atac_file()
    adata = sciv.fl.read_sc_atac(os.path.join(sciv.ul.project_cache_path, "GSE139369_ELM_sim_snapATAC2.h5ad"))

1.2.2 PoissonVI process
*****************************

.. code-block:: python

    # These parameters (cell_rate, peak_rate) speed up execution for testing purposes.
    # Remove or adjust them appropriately in production use cases.
    filter_data(adata, cell_rate=0.05, peak_rate=0.001)
    da_peaks = poisson_vi(adata, model_dir="./result/poisson_vi")


1.2.4 Overlap process
*****************************

.. code-block:: python

    overlap_adata: AnnData = overlap_sum(adata, variants, trait_info)


1.2.5 Initialize TRS process
*****************************

.. code-block:: python

    init_score: AnnData = calculate_init_score_weight(
        adata=adata,
        da_peaks_adata=da_peaks,
        overlap_adata=overlap_adata
    )

1.2.6 Cell cell similarity network
*****************************

.. code-block:: python

    cc_data = obtain_cell_cell_network(adata)

1.2.7 Weighted random walk
*****************************

.. code-block:: python

    random_walk: RandomWalk = RandomWalk(cc_adata=cc_data, init_status=init_score)
    random_walk.run_core()
    random_walk.run_enrichment()
    trs = random_walk.trs_adata
