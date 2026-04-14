1.	SCIV usage
=========================


1.1 Install
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell

    conda create --name sciv python=3.10
    conda activate sciv
    pip install sciv


1.2 SCIV execution process
^^^^^^^^^^^^^^^^^^^^^^^^^^

1.2.1 Download scATAC-seq sample data
****************************

Download PBMC case file： `GSE139369_ELM_sim_snapATAC2.h5ad <https://bio.liclab.net/scvmap_static/sciv/GSE139369_ELM_sim_snapATAC2.h5ad>`_

.. code-block:: python

    import sciv
    adata = sciv.dl.read_sc_atac_file()

1.2.2 Download trait example data
****************************

Download the fine mapping results for monocytes, red blood cells, CD4+ and CD8+ T cells

.. code-block:: python

    variants, trait_info = sciv.dl.read_trait_file()

1.2.3 Run SCIV
****************************

Create Python file

.. code-block:: shell

    mkdir -p /project/sciv/code/GSE139369_ELM/
    cd /project/sciv/code/GSE139369_ELM/
    touch sciv_pbmc.py

The file content is as follows

.. code-block:: python

    # -*- coding: UTF-8 -*-

    import os.path
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
            model_dir=os.path.join(save_path, "poisson_vi"),
            is_file_exist_loading=True
        )

        print(trs)

Executable the file

.. code-block:: shell

    python3 sciv_pbmc.py

The output log information is as follows

.. code-block:: shell

    python3 sciv_pbmc.py
