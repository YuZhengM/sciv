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
            save_path="./result",
            model_dir="./result/poisson_vi",
            is_file_exist_loading=True
        )

        print(trs)

(3) Executable the file:

.. code-block:: shell

    python3 sciv_pbmc.py


1.2 SCIV execution process for each step
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



