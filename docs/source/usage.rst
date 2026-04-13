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

1.2.1 Download scATAC seq sample data
****************************

Download PBMC case file： `GSM6793454_sc_atac_snapATAC2.h5ad <https://bio.liclab.net/scvmap_static/download/scatac/GSM6793454_sc_atac_snapATAC2.h5ad>`_

.. code-block:: shell

    mkdir -p /project/sciv/input/scATAC/GSM6793454
    cd /project/sciv/input/scATAC/GSM6793454
    wget https://bio.liclab.net/scvmap_static/download/scatac/GSM6793454_sc_atac_snapATAC2.h5ad

1.2.2 Download trait example data
****************************

Download the fine mapping results for monocytes, B cells, CD4+ and CD8+ T cells

.. code-block:: shell

    mkdir -p /project/sciv/input/trait/GSM6793454
    cd /project/sciv/input/trait/GSM6793454
    wget https://bio.liclab.net/scvmap_static/sciv/download_example_traits.sh
    chmod +x download_example_traits.sh
    sh download_example_traits.sh
    rm -rf download_example_traits.sh


1.2.3 Run SCIV
****************************

Create Python file

.. code-block:: shell

    mkdir -p /project/sciv/code/GSM6793454/
    cd /project/sciv/code/GSM6793454/
    touch sciv_pbmc.py

The file content is as follows

.. code-block:: python

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

Executable the file

.. code-block:: shell

    python3 sciv_pbmc.py

The output log information is as follows

.. code-block:: shell

    python3 sciv_pbmc.py
