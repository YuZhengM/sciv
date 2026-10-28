# -*- coding: UTF-8 -*-

import os
import time
import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd

from anndata import AnnData
from torch.cuda import OutOfMemoryError
from lightning.pytorch.strategies import DDPStrategy

from .. import util as ul
from ..tool import umap, tsne, leiden
from ..util import path, check_gpu_availability

__name__: str = "preprocessing_scvi"


def poisson_vi(
    adata: AnnData,
    max_epochs: int = 500,
    lr: float = 1e-4,
    batch_size: int = 128,
    eps: float = 1e-08,
    early_stopping: bool = True,
    early_stopping_patience: int = 50,
    strategy: Union[str, DDPStrategy] = DDPStrategy(start_method="spawn", find_unused_parameters=True),
    batch_key: Optional[str] = None,
    resolution: float = 0.5,
    dp_delta: float = 0.05,
    latent_name: str = "latent",
    model_dir: Optional[path] = None,
    train_kwargs: Optional[dict] = None,
    diff_peak_kwargs: Optional[dict] = None
) -> AnnData:
    """
    PoissonVI processing of the data results in the current sample representation 
    and peak difference data after Leiden clustering.

    Parameters
    ----------
    adata : AnnData
        Input data to be processed.
    max_epochs : int, default 500
        The maximum number of epochs for PoissonVI training.
    lr : float, default 1e-4
        Learning rate for optimization.
    batch_size : int, default 128
        Minibatch size to use during training.
    eps : float, default 1e-08
        Optimizer epsilon.
    early_stopping : bool, default True
        Whether to perform early stopping with respect to the validation set.
    early_stopping_patience : int, default 50
        How many epochs to wait for improvement before early stopping.
    strategy : str, default "ddp_notebook_find_unused_parameters_true"
        DDP strategy.
    batch_key : str, optional
        Batch information in scATAC-seq data.
    resolution : float, default 0.5
        Resolution of the Leiden clustering.
    dp_delta : float, default 0.05
        Empirical effect size threshold for PeakVI method in differential analysis.
    latent_name : str, default "latent"
        The name of latent representation.
    model_dir : str, optional
        The folder name for saving the trained model.
    train_kwargs : dict, optional
    diff_peak_kwargs : dict, optional

    Returns
    -------
    AnnData
        Differential peak data of clustering types.
    """
    ul.log(__name__).info("Start PoissonVI")

    start_time = time.perf_counter()

    import scvi

    if resolution <= 0:
        ul.log(__name__).error("The parameter `resolution` must be greater than zero.")
        raise ValueError("The parameter `resolution` must be greater than zero.")

    if dp_delta <= 0:
        ul.log(__name__).error("The parameter `dp_delta` must be greater than zero.")
        raise ValueError("The parameter `dp_delta` must be greater than zero.")

    if batch_key is not None and batch_key not in adata.obs.columns:
        ul.log(__name__).error(
            f"The cells information {adata.obs.columns} in data `adata` must include the {batch_key} column."
        )
        raise ValueError(
            f"The cells information {adata.obs.columns} in data `adata` must include the {batch_key} column."
        )

    # PoissonVI, Binarization
    ul.log(__name__).info("Calculate fragment counts matrix.")
    scvi.data.reads_to_fragments(adata)

    if train_kwargs is not None and isinstance(train_kwargs, dict):

        if "check_val_every_n_epoch" not in diff_peak_kwargs:
            diff_peak_kwargs["check_val_every_n_epoch"] = 1

        if "accelerator" not in diff_peak_kwargs:
            diff_peak_kwargs["accelerator"] = "gpu"

        if "devices" not in diff_peak_kwargs:
            diff_peak_kwargs["devices"] = -1

    def __train__():

        # PoissonVI
        scvi.external.POISSONVI.setup_anndata(adata, layer="fragments", batch_key=batch_key)
        _model_ = scvi.external.POISSONVI(adata)

        if check_gpu_availability():

            try:
                data_splitter_kwargs = {"drop_dataset_tail": True, "drop_last": False}

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _model_.train(
                        max_epochs=int(max_epochs),
                        datasplitter_kwargs=data_splitter_kwargs,
                        strategy=strategy,
                        lr=lr,
                        batch_size=int(batch_size),
                        eps=eps,
                        early_stopping=early_stopping,
                        early_stopping_patience=int(early_stopping_patience),
                        **train_kwargs
                    )
            except Exception as ex:

                try:
                    ul.log(__name__).warning(f"Multiple GPUs failed to run, attempting to run program with parameter "
                                             f"devices='auto'.\n {ex}")
                    train_kwargs.update({"device": "auto"})

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        _model_.train(
                            max_epochs=int(max_epochs),
                            lr=lr,
                            batch_size=int(batch_size),
                            eps=eps,
                            early_stopping=early_stopping,
                            early_stopping_patience=int(early_stopping_patience),
                            **train_kwargs
                        )
                except Exception as exc:
                    ul.log(__name__).warning(f"GPU failed to run, try to switch to CPU running.\n {exc}")
                    train_kwargs.update({"accelerator": "cpu"})

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        _model_.to_device('cpu')
                        _model_.train(
                            max_epochs=int(max_epochs),
                            lr=lr,
                            batch_size=int(batch_size),
                            eps=eps,
                            early_stopping=early_stopping,
                            early_stopping_patience=int(early_stopping_patience),
                            **train_kwargs
                        )
        else:

            train_kwargs.update({"device": "auto", "accelerator": "cpu"})

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _model_.to_device('cpu')
                _model_.train(
                    max_epochs=int(max_epochs),
                    lr=lr,
                    batch_size=int(batch_size),
                    eps=eps,
                    early_stopping=early_stopping,
                    early_stopping_patience=int(early_stopping_patience),
                    **train_kwargs
                )

        return _model_

    if model_dir is not None:
        if os.path.exists(os.path.join(model_dir, "model.pt")):
            ul.log(__name__).info(f"Due to the existence of file `model.pt`, it is loaded by default.")

            try:
                model = scvi.external.POISSONVI.load(
                    model_dir,
                    adata=adata,
                    accelerator="gpu" if check_gpu_availability() else "auto"
                )
            except OutOfMemoryError as ome:
                ul.log(__name__).warning(f"GPU failed to run, try to switch to CPU running.\n {ome}")

                try:
                    model = scvi.external.POISSONVI.load(model_dir, adata=adata, accelerator="cpu")
                except Exception as e:
                    ul.log(__name__).error(
                        f"File `model.pt` failed to load, you can execute `Poisson VI` again by deleting file \
                         `model.pt` ({model_dir}/model.pt).\n {e}"
                    )
                    raise ValueError(
                        f"File `model.pt` failed to load, you can execute `Poisson VI` again by deleting file \
                         `model.pt` ({model_dir}/model.pt)."
                    )
        else:
            ul.file_method(__name__).makedirs(model_dir)
            model = __train__()

            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                model.save(model_dir, overwrite=True)
    else:
        model = __train__()

    # latent space
    ul.log(__name__).info("Obtain latent representation matrix.")
    latent = model.get_latent_representation()
    adata.obsm[latent_name] = latent

    if "clusters" in adata.obs.columns:
        ul.log(__name__).warning(
            "Due to the original inclusion of the `clusters` column, the original `clusters` column name has "
            "been changed to `clusters_x`."
        )
        adata.obs["clusters_x"] = adata.obs["clusters"]

    ul.log(__name__).info(f"Perform KNN and leiden clustering.")
    leiden(adata, latent_name=latent_name, key_added="clusters", resolution=resolution)
    adata.obs["clusters"] = adata.obs["clusters"].astype(str)

    try:
        ul.log(__name__).info(f"UMAP")
        data_umap = umap(adata.obsm[latent_name])
        adata.obsm["umap"] = data_umap
        adata.obs["latent_umap1"] = data_umap[:, 0]
        adata.obs["latent_umap2"] = data_umap[:, 1]
    except Exception as e:
        ul.log(__name__).warning(
            f"UMAP error, your system does not support it, but it does not affect the process. "
            f"Continue with execution: {e}"
        )

    try:
        ul.log(__name__).info(f"t-SNE")
        data_tsne = tsne(adata.obsm[latent_name])
        adata.obsm["tsne"] = data_tsne
        adata.obs["latent_tsne1"] = data_tsne[:, 0]
        adata.obs["latent_tsne2"] = data_tsne[:, 1]
    except Exception as e:
        ul.log(__name__).warning(
            f"TSNE error, your system does not support it, but it does not affect the process. "
            f"Continue with execution: {e}"
        )

    clusters_list = list(set(adata.obs["clusters"]))
    clusters_list.sort()

    if diff_peak_kwargs is not None and isinstance(diff_peak_kwargs, dict):

        if "mode" not in diff_peak_kwargs:
            diff_peak_kwargs["mode"] = "vanilla"

        if "two_sided" not in diff_peak_kwargs:
            diff_peak_kwargs["two_sided"] = False

    params = {
        "max_epochs": max_epochs,
        "lr": lr,
        "batch_size": batch_size,
        "eps": eps,
        "model_dir": model_dir,
        "early_stopping_patience": early_stopping_patience,
        "strategy": strategy if isinstance(strategy, str) else {
            "_ddp_kwargs": strategy._ddp_kwargs,
            "_start_method": strategy._start_method,
            "_timeout": strategy._timeout.seconds
        },
        "batch_key": batch_key,
        "resolution": resolution,
        "cluster_size": len(clusters_list),
        "dp_delta": dp_delta,
        "latent_name": latent_name,
        "train_kwargs": train_kwargs,
        "diff_peak_kwargs": diff_peak_kwargs
    }

    adata.uns["poisson_vi"] = params
    peaks_info = adata.var.copy()
    peaks_info["index"] = peaks_info.index
    peaks_info.rename_axis("peak_index", inplace=True)

    matrix_bf = np.ones((len(clusters_list), adata.shape[1]))
    matrix_ep1 = np.ones((len(clusters_list), adata.shape[1]))
    matrix_ee = np.ones((len(clusters_list), adata.shape[1]))

    if len(clusters_list) == 1:
        ul.log(__name__).warning(
            f"The number of clusters is one, ignoring cluster type correction. It is recommended to increase the "
            f"`resolution` value to make it effective."
        )
    else:
        # differential peak
        da_peaks_all: dict = {}

        for cluster in clusters_list:
            ul.log(__name__).info(f"Start difference peak: {cluster}/({', '.join(clusters_list)})")
            clusters_all = clusters_list.copy()
            clusters_all.remove(cluster)

            # differential peak
            try:
                da_peaks = model.differential_accessibility(
                    adata, groupby="clusters", delta=dp_delta, group1=cluster, **diff_peak_kwargs
                )
            except Exception as e:
                ul.log(__name__).warning(f"GPU failed to run, try to switch to CPU running.\n {e}")
                # PyTorch uses CPU
                model.to_device('cpu')
                da_peaks = model.differential_accessibility(
                    adata, groupby="clusters", delta=dp_delta, group1=cluster, **diff_peak_kwargs
                )

            da_peaks_all.update({cluster: da_peaks})

        adata.uns["da_peaks"] = da_peaks_all

        for i in range(len(clusters_list)):
            cluster_info = da_peaks_all[clusters_list[i]]
            cluster_info["index"] = cluster_info.index
            cluster_info.rename_axis("cluster_index", inplace=True)
            cluster_info = pd.merge(left=peaks_info, right=cluster_info, left_on="index", right_on="index", how="left")
            matrix_bf[i, :] = cluster_info["bayes_factor"]
            matrix_ep1[i, :] = cluster_info["emp_prob1"]
            matrix_ee[i, :] = cluster_info["emp_effect"]

    ul.log(__name__).info("End PoissonVI")

    obs = pd.DataFrame(clusters_list, columns=["id"])
    obs.index = obs["id"].astype(str)

    da_peaks_adata = AnnData(matrix_ee, obs=obs, var=adata.var)
    da_peaks_adata.layers["bayes_factor"] = matrix_bf
    da_peaks_adata.layers["emp_prob1"] = matrix_ep1
    da_peaks_adata.uns["latent_name"] = latent_name
    da_peaks_adata.uns["dp_delta"] = dp_delta
    da_peaks_adata.uns["params"] = params
    da_peaks_adata.uns["elapsed_time"] = time.perf_counter() - start_time

    adata.uns["step"] = 1

    return da_peaks_adata
