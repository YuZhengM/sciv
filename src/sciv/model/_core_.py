# -*- coding: UTF-8 -*-

import os.path
import time
from typing import Optional, Union, Literal

import numpy as np

import pandas as pd
from anndata import AnnData
from pandas import DataFrame

from .. import util as ul
from ..tool import RandomWalk, overlap_sum, obtain_cell_cell_network, calculate_init_score_weight

from ..file import save_h5ad, save_pkl, read_h5ad, read_pkl
from ..preprocessing import filter_data, poisson_vi
from ..util import path, enrichment_optional, to_dense, collection, to_sparse, difference_peak_optional, project_name

__name__: str = "model_core"


def _run_random_walk_(random_walk: RandomWalk, is_ablation: bool, is_simple: bool) -> AnnData:
    """
    Execute random walk algorithm and optional ablation experiments.

    This function runs the core random walk algorithm and enrichment analysis,
    optionally running various ablation experiments when is_ablation is True
    and is_simple is False.

    Parameters
    ----------
    random_walk : RandomWalk
        RandomWalk instance containing all necessary data and methods
    is_ablation : bool
        Whether to run ablation experiments (only effective when is_simple=False)
    is_simple : bool
        Whether to skip intermediate steps and ablation experiments

    Returns
    -------
    AnnData
        AnnData object containing TRS (Trait Relevance Score) results
    """
    start_time = time.perf_counter()

    if not random_walk.is_run_core:
        random_walk.run_core()

    if not random_walk.is_run_enrichment:
        random_walk.run_enrichment()

    if is_ablation and not is_simple:
        if not random_walk.is_benchmark:
            random_walk.run_benchmark()

        if not random_walk.is_run_ablation_ncw:
            random_walk.run_ablation_ncw()

        if not random_walk.is_run_ablation_nsw:
            random_walk.run_ablation_nsw()

        if not random_walk.is_run_ablation_ncsw:
            random_walk.run_ablation_ncsw()

        if not random_walk.is_run_ablation_m_knn:
            random_walk.run_ablation_m_knn()

        if not random_walk.is_run_en_ablation_ncw:
            random_walk.run_en_ablation_ncw()

        if not random_walk.is_run_en_ablation_nsw:
            random_walk.run_en_ablation_nsw()

        if not random_walk.is_run_en_ablation_ncsw:
            random_walk.run_en_ablation_ncsw()

        if not random_walk.is_run_en_ablation_m_knn:
            random_walk.run_en_ablation_m_knn()

    random_walk.elapsed_time += time.perf_counter() - start_time

    return random_walk.trs_adata


def core(
    adata: AnnData,
    variants: dict,
    trait_info: DataFrame,
    cell_rate: Optional[float] = None,
    peak_rate: Optional[float] = None,
    max_epochs: int = 500,
    lr: float = 1e-5,
    batch_size: int = 128,
    eps: float = 1e-08,
    early_stopping: bool = True,
    early_stopping_patience: int = 50,
    strategy: str = "ddp_notebook_find_unused_parameters_true",
    batch_key: Optional[str] = None,
    resolution: float = 0.5,
    k: int = 30,
    or_k: int = 10,
    weight: float = 0.5,
    kernel: Literal["laplacian", "gaussian"] = "gaussian",
    local_k: int = 10,
    kernel_gamma: Optional[Union[float, str, collection]] = None,
    epsilon: float = 1e-05,
    max_steps: int = 300,
    gamma: float = 0.05,
    enrichment_gamma: float = 0.05,
    p: int = 2,
    n_jobs: int = -1,
    min_seed_cell_rate: float = 0.01,
    max_seed_cell_rate: float = 0.05,
    credible_threshold: float = 0,
    diff_peak_value: difference_peak_optional = 'emp_effect',
    enrichment_threshold: Union[enrichment_optional, float] = 'golden',
    is_ablation: bool = False,
    model_dir: Optional[path] = None,
    save_path: Optional[path] = None,
    is_simple: bool = True,
    is_save_random_walk_model: bool = False,
    is_file_exist_loading: bool = False,
    filename_dict: Optional[dict] = None,
    block_size: int = -1
) -> AnnData:
    """
    The core algorithm of sciv includes the flow of all algorithms, as well as drawing and saving data.
    In the entire algorithm, the samples are in the row position, and the traits or diseases are in the column position,
        while ensuring that there is no interaction between the traits or diseases,
        ensuring the stability of the results;

    Meaning of main variables:
        1. `overlap_adata`, (obs: peaks, var: traits/diseases) Peaks-traits/diseases data obtained by overlaying variant
         data with peaks.
        2. `da_peaks`, (obs: clusters (Leiden), var: peaks) Differential peak data of cell clustering, used for weight
         correction of cells.
        3. `init_score`, (obs: cells, var: traits/diseases) This is the initial TRS data.
        4. `cc_data`, (obs: cells, var: cells) Cell similarity data.
        5. `random_walk`, RandomWalk class.
        6. `trs`, (obs: cells, var: traits/diseases) This is the final TRS data.

    Parameters
    ----------
    adata : AnnData
        scATAC-seq data.
    variants : dict
        Variant data. This data is recommended to be obtained by executing the `fl.read_variants` method.
    trait_info : DataFrame
        Variant annotation file information.
    cell_rate : Optional[float], default None
        Removing the percentage of cell count in total cell count only takes effect when the min_cells
        parameter is None.
    peak_rate : Optional[float], default None
        Removing the percentage of peak count in total peak count only takes effect when the min_peaks
        parameter is None.
    max_epochs : int, default 500
        The maximum number of epochs for PoissonVI training.
    lr : float, default 1e-5
        Learning rate for optimization.
    batch_size : int, default 128
        Minibatch size to use during training.
    eps : float, default 1e-08
        Optimizer eps.
    early_stopping : bool, default True
        Whether to perform early stopping with respect to the validation set.
    early_stopping_patience : int, default 50
        How many epochs to wait for improvement before early stopping.
    strategy : str, default "ddp_notebook_find_unused_parameters_true"
        DDP strategy.
    batch_key : Optional[str], default None
        Batch information in scATAC-seq data.
    resolution : float, default 0.5
        Resolution of the Leiden Cluster. The recommended values are any one of 0.4, 0.9, 1.3, 1.5.
    k : int, default 30
        When building an mKNN network, the number of nodes connected by each node (and operation).
    or_k : int, default 10
        When building an mKNN network, the number of nodes connected by each node (or operation).
    weight : float, default 0.5
        The weight of interactions or operations.
    kernel : Literal["laplacian", "gaussian"], default "gaussian"
        Determine the kernel function to be used.
    local_k : int, default 10
        Determining the number of neighbors for the adaptive kernel.
    kernel_gamma : Optional[Union[float, str, collection]], default None
        When the value of `kernel` is "laplacian", if it is None, then it is the reciprocal of the
        latent representation dimension of the cell. When the value of `kernel` is "gaussian", if it is None, then it
        defaults to an adaptive value obtained through local information of the parameter `local_k`. Otherwise, it
        should be strictly positive.
    epsilon : float, default 1e-05
        Conditions for stopping in random walk.
    max_steps : int, default 300
        Maximum number of steps in a random walk with restart.
    gamma : float, default 0.05
        Reset weight for random walk.
    enrichment_gamma : float, default 0.05
        Reset weight for random walk for enrichment.
    p : int, default 2
        Distance used for loss {1: Manhattan distance, 2: Euclidean distance}.
    n_jobs : int, default -1
        The maximum number of concurrently running jobs.
    min_seed_cell_rate : float, default 0.01
        The minimum percentage of seed cells in all cells.
    max_seed_cell_rate : float, default 0.05
        The maximum percentage of seed cells in all cells.
    credible_threshold : float, default 0
        The threshold for determining the credibility of enriched cells in the context of
        enrichment, i.e. the threshold for judging enriched cells.
    diff_peak_value : difference_peak_optional, default 'emp_effect'
        Specify the correction value in peak correction of clustering type differences.
        {'emp_effect', 'bayes_factor', 'emp_prob1'}
    enrichment_threshold : Union[enrichment_optional, float], default 'golden'
        Only by setting a threshold for the standardized output TRS can a portion of the
        enrichment results be obtained. Parameters support string types {'golden', 'half', 'e', 'pi', 'none'},
        or valid floating-point types within the range of (0, log1p(1)).
    is_ablation : bool, default False
        True represents obtaining the results of the ablation experiment. This parameter is limited by
        the `is_simple` parameter, and its effectiveness requires setting `is_simple` to `False`.
    model_dir : Optional[path], default None
        The folder name saved by the training module.
        It is worth noting that if the training model file (`model.pt`) exists in this path, it will be automatically
        read and skip the training of `PoissonVI` model.
    save_path : Optional[path], default None
        Save path for process files and result files.
    is_simple : bool, default True
        True represents not adding unnecessary intermediate variables, only adding the final result.
        It is worth noting that when set to `True`, the `is_ablation` parameter will become invalid, and when set to
        `False`, `is_ablation` will only take effect.
    is_save_random_walk_model : bool, default False
        Default to `False`, do not save random walk model. When setting `True`, please
        ensure sufficient storage as the saved `pkl` file is relatively large.
    is_file_exist_loading : bool, default False
        By default, the file will be overwritten. When set to `True`, if the file exists, the
        process will be skipped and the file will be directly read as the result.
    filename_dict : Optional[dict], default None
        The name of the file that exists.
        default: {
            "sc_atac": "sc_atac.h5ad",
            "da_peaks": "da_peaks.h5ad",
            "atac_overlap": "atac_overlap.h5ad",
            "init_score": "init_score.h5ad",
            "cc_data": "cc_data.h5ad",
            "random_walk": "random_walk.h5ad",
            "trs": "trs.h5ad"
        }
    block_size : int
        The size of the segmentation stored in block wise matrix multiplication.
        By sacrificing time and space to reduce memory consumption to a certain extent.
        If the value is less than or equal to zero, no block operation will be performed.
    
    Returns
    -------
    AnnData
        AnnData object containing TRS (Trait Relevance Score) results.
        (obs: cells, var: traits/diseases) This is the final TRS data.
    """
    # start time
    start_time = time.perf_counter()

    if len(variants.keys()) == 0:
        ul.log(__name__).error("The number of mutations is empty.")
        raise ValueError("The number of mutations is empty.")

    _trait_count_ = trait_info.shape[0]

    if len(variants.keys()) != _trait_count_:
        ul.log(__name__).error(
            "The parameters `variants` and `trait_info` are inconsistent. "
            "These two parameters can be obtained using method `fl.read_variants`."
        )
        raise ValueError(
            "The parameters `variants` and `trait_info` are inconsistent. "
            "These two parameters can be obtained using method `fl.read_variants`."
        )

    if adata.shape[0] == 0:
        ul.log(__name__).error("The scATAC-seq data is empty.")
        raise ValueError("The scATAC-seq data is empty.")

    peak_columns: list = list(adata.var.columns)

    if "chr" not in peak_columns or "start" not in peak_columns or "end" not in peak_columns:
        ul.log(__name__).error(
            f"The peaks information {peak_columns} in data `adata` must include three columns: "
            f"`chr`, `start` and `end`. (It is recommended to use the `fl.read_sc_atac` method.)"
        )
        raise ValueError(
            f"The peaks information {peak_columns} in data `adata` must include three columns: "
            f"`chr`, `start` and `end`. (It is recommended to use the `fl.read_sc_atac` method.)"
        )

    if batch_key is not None and batch_key not in adata.obs.columns:
        ul.log(__name__).error(
            f"The cells information {adata.obs.columns} in data `adata` must include the {batch_key} column."
        )
        raise ValueError(
            f"The cells information {adata.obs.columns} in data `adata` must include the {batch_key} column."
        )

    if cell_rate is not None:

        if cell_rate <= 0 or cell_rate >= 1:
            ul.log(__name__).error("The parameter of `cell_rate` should be between 0 and 1.")
            raise ValueError("The parameter of `cell_rate` should be between 0 and 1.")

    if peak_rate is not None:

        if peak_rate <= 0 or peak_rate >= 1:
            ul.log(__name__).error("The parameter of `peak_rate` should be between 0 and 1.")
            raise ValueError("The parameter of `peak_rate` should be between 0 and 1.")

    if resolution <= 0:
        ul.log(__name__).error("The parameter `resolution` must be greater than zero.")
        raise ValueError("The parameter `resolution` must be greater than zero.")

    if k <= 0:
        ul.log(__name__).error("The `k` parameter must be a natural number greater than 0.")
        raise ValueError("The `k` parameter must be a natural number greater than 0.")

    if or_k <= 0:
        ul.log(__name__).error("The `or_k` parameter must be a natural number greater than 0.")
        raise ValueError("The `or_k` parameter must be a natural number greater than 0.")

    if k < or_k:
        ul.log(__name__).warning(
            "The parameter value of `or_k` is greater than the parameter value of `k`, "
            "which is highly likely to result in poor performance."
        )

    if local_k <= 0:
        ul.log(__name__).error("The `local_k` parameter must be a natural number greater than 0.")
        raise ValueError("The `local_k` parameter must be a natural number greater than 0.")

    if kernel not in ["laplacian", "gaussian"]:
        ul.log(__name__).error("Parameter `kernel` only supports two values, `laplacian` and `gaussian`.")
        raise ValueError("Parameter `kernel` only supports two values, `laplacian` and `gaussian`.")

    if weight < 0 or weight > 1:
        ul.log(__name__).error("The parameter of `weight` should be between 0 and 1.")
        raise ValueError("The parameter of `weight` should be between 0 and 1.")

    if gamma < 0 or gamma > 1:
        ul.log(__name__).error("The parameter of `gamma` should be between 0 and 1.")
        raise ValueError("The parameter of `gamma` should be between 0 and 1.")

    if enrichment_gamma < 0 or enrichment_gamma > 1:
        ul.log(__name__).error("The parameter of `enrichment_gamma` should be between 0 and 1.")
        raise ValueError("The parameter of `enrichment_gamma` should be between 0 and 1.")

    if max_steps <= 0:
        ul.log(__name__).error("The `max_steps` parameter must be a natural number greater than 0.")
        raise ValueError("The `max_steps` parameter must be a natural number greater than 0.")

    if min_seed_cell_rate < 0 or min_seed_cell_rate > 1:
        ul.log(__name__).error("The parameter of `min_seed_cell_rate` should be between 0 and 1.")
        raise ValueError("The parameter of `min_seed_cell_rate` should be between 0 and 1.")

    if max_seed_cell_rate < 0 or max_seed_cell_rate > 1:
        ul.log(__name__).error("The parameter of `max_seed_cell_rate` should be between 0 and 1.")
        raise ValueError("The parameter of `max_seed_cell_rate` should be between 0 and 1.")

    if epsilon > 0.1:
        ul.log(__name__).warning(
            f"Excessive value of parameter `epsilon`=({epsilon}) can lead to "
            f"incorrect iteration and poor enrichment effect."
        )
    elif epsilon <= 0:
        ul.log(__name__).error("The parameter of `epsilon` must be greater than zero.")
        raise ValueError("The parameter of `epsilon` must be greater than zero.")

    if p <= 0:
        ul.log(__name__).error("The `p` parameter must be a natural number greater than 0.")
        raise ValueError("The `p` parameter must be a natural number greater than 0.")
    elif p > 3:
        ul.log(__name__).warning("Suggested value for `p` is 1 or 2.")

    if isinstance(enrichment_threshold, float):

        if enrichment_threshold <= 0 or enrichment_threshold >= np.log1p(1):
            ul.log(__name__).warning(
                "The `enrichment_threshold` parameter is not set within the range of (0, log1p(1)), this parameter "
                "will become invalid."
            )
            ul.log(__name__).warning(
                "It is recommended to set the `enrichment_threshold` parameter to the 'golden' value."
            )

    elif enrichment_threshold not in ["golden", "half", "e", "pi", "none"]:
        ul.log(__name__).error(
            "Invalid enrichment settings. The string type in the `enrichment_threshold` parameter only supports the "
            "following parameter 'golden', 'half', 'e', 'pi', 'none',  Alternatively, input a floating-point type "
            "value within the range of (0, log1p(1))"
        )
        raise ValueError(
            "Invalid enrichment settings. The string type in the `enrichment_threshold` parameter only supports the "
            "following parameter 'golden', 'half', 'e', 'pi', 'none',  Alternatively, input a floating-point type "
            "value within the range of (0, log1p(1))"
        )

    if diff_peak_value not in ['emp_effect', 'bayes_factor', 'emp_prob1', 'all']:
        ul.log(__name__).error(
            "The `diff_peak_value` parameter only supports one of the "
            "{'emp_effect', 'bayes_factor', 'emp_prob1', 'all'} values."
        )
        raise ValueError(
            "The `diff_peak_value` parameter only supports one of the "
            "{'emp_effect', 'bayes_factor', 'emp_prob1', 'all'} values."
        )

    # parameter information
    params: dict = {
        "cell_rate": cell_rate,
        "peak_rate": peak_rate,
        "max_epochs": int(max_epochs),
        "lr": lr,
        "batch_size": batch_size,
        "eps": eps,
        "early_stopping": early_stopping,
        "early_stopping_patience": early_stopping_patience,
        "strategy": strategy,
        "batch_key": batch_key,
        "resolution": resolution,
        "k": k,
        "or_k": or_k,
        "weight": weight,
        "kernel": kernel,
        "local_k": local_k,
        "kernel_gamma": kernel_gamma,
        "epsilon": epsilon,
        "gamma": gamma,
        "enrichment_gamma": enrichment_gamma,
        "max_steps": max_steps,
        "p": p,
        "n_jobs": n_jobs,
        "min_seed_cell_rate": min_seed_cell_rate,
        "max_seed_cell_rate": max_seed_cell_rate,
        "credible_threshold": credible_threshold,
        "enrichment_threshold": enrichment_threshold,
        "diff_peak_value": diff_peak_value,
        "is_ablation": is_ablation,
        "model_dir": str(model_dir),
        "save_path": str(save_path),
        "is_simple": is_simple,
        "is_save_random_walk_model": is_save_random_walk_model,
        "is_file_exist_loading": is_file_exist_loading,
        "filename_dict": filename_dict,
        "block_size": block_size
    }

    """
    Save information
    """

    if save_path is not None:
        save_path = str(save_path)
        ul.file_method(__name__).makedirs(save_path)
    else:
        is_file_exist_loading = False

    if filename_dict is None:
        filename_dict = {
            "sc_atac": "sc_atac.h5ad",
            "da_peaks": "da_peaks.h5ad",
            "atac_overlap": "atac_overlap.h5ad",
            "init_score": "init_score.h5ad",
            "cc_data": "cc_data.h5ad",
            "random_walk": "random_walk.h5ad",
            "trs": "trs.h5ad"
        }

    def _get_file_(_id_: str) -> str:
        return os.path.join(save_path, str(filename_dict.get(_id_, f"{_id_}.h5ad"))) if save_path else None

    # Assign a name to the formed document
    adata_save_file = _get_file_("sc_atac")
    da_peaks_save_file = _get_file_("da_peaks")
    atac_overlap_save_file = _get_file_("atac_overlap")
    init_score_save_file = _get_file_("init_score")
    cc_data_save_file = _get_file_("cc_data")
    random_walk_save_file = _get_file_("random_walk")
    trs_save_file = _get_file_("trs")

    """
    1. Filter scATAC-seq data, PoissonVI
    """

    adata_is_read: bool = False
    da_peaks_is_read: bool = False

    if is_file_exist_loading:

        if os.path.exists(adata_save_file):
            adata = read_h5ad(adata_save_file)
            adata_is_read = True

            if "step" not in adata.uns.keys():
                filter_data(adata, cell_rate=cell_rate, peak_rate=peak_rate)
                adata_is_read = False
        else:
            filter_data(adata, cell_rate=cell_rate, peak_rate=peak_rate)

        if os.path.exists(da_peaks_save_file) and adata.uns["step"] == 1:
            da_peaks = read_h5ad(da_peaks_save_file)
            da_peaks_is_read = True
        else:
            # PoissonVI
            da_peaks = poisson_vi(
                adata,
                max_epochs=max_epochs,
                lr=lr,
                batch_size=batch_size,
                eps=eps,
                early_stopping=early_stopping,
                early_stopping_patience=early_stopping_patience,
                strategy=strategy,
                batch_key=batch_key,
                resolution=resolution,
                model_dir=model_dir
            )

    else:
        filter_data(adata, cell_rate=cell_rate, peak_rate=peak_rate)

        da_peaks = poisson_vi(
            adata,
            max_epochs=max_epochs,
            lr=lr,
            batch_size=batch_size,
            eps=eps,
            early_stopping=early_stopping,
            early_stopping_patience=early_stopping_patience,
            strategy=strategy,
            batch_key=batch_key,
            resolution=resolution,
            model_dir=model_dir
        )

    poisson_vi_time = adata.uns["elapsed_time"] + da_peaks.uns["elapsed_time"]

    if save_path is not None:

        if not adata_is_read:
            save_h5ad(adata, file=adata_save_file)

        if not da_peaks_is_read:
            save_h5ad(da_peaks, file=da_peaks_save_file)

    """
    2. Overlap regional data and mutation data and sum the PP values of all mutations
       in a region as the values for that region
    """

    # Determine whether it is necessary to read the file
    overlap_is_read: bool = is_file_exist_loading and os.path.exists(atac_overlap_save_file)

    if overlap_is_read:
        overlap_adata: AnnData = read_h5ad(atac_overlap_save_file)

        if overlap_adata.var.shape[0] != _trait_count_:
            ul.log(__name__).warning(
                f"The number of diseases read from file `atac_overlap.h5ad` are inconsistent with the input ({overlap_adata.var.shape[0]} != {_trait_count_}). "
                f"Please check and verify. If the verification is not as expected, file `atac_overlap.h5ad` needs to be moved or deleted."
            )

    else:
        overlap_adata: AnnData = overlap_sum(adata, variants, trait_info, n_jobs=n_jobs)

    del variants, trait_info

    overlap_time = overlap_adata.uns["elapsed_time"]

    if save_path is not None and not overlap_is_read:
        save_h5ad(overlap_adata, file=atac_overlap_save_file)

    """
    3. Calculate the initial trait relevance scores for each cell
    """

    init_score_is_read: bool = is_file_exist_loading and os.path.exists(init_score_save_file)

    if init_score_is_read:
        init_score: AnnData = read_h5ad(init_score_save_file)

        if init_score.var.shape[0] != _trait_count_:
            ul.log(__name__).warning(
                f"The number of diseases read from file `init_score.h5ad` are inconsistent with the input ({init_score.var.shape[0]} != {_trait_count_}). "
                f"Please check and verify. If the verification is not as expected, file `init_score.h5ad` needs to be moved or deleted."
            )

    else:
        # intermediate score data, integration data
        init_score: AnnData = calculate_init_score_weight(
            adata=adata,
            da_peaks_adata=da_peaks,
            overlap_adata=overlap_adata,
            diff_peak_value=diff_peak_value,
            is_simple=is_simple,
            block_size=block_size
        )

    del da_peaks, overlap_adata

    init_score_time = init_score.uns["elapsed_time"]

    if save_path is not None and not init_score_is_read:
        save_h5ad(init_score, file=init_score_save_file)

    """
    4. Calculate cell-cell correlation. Building a network between cells.
    """

    cc_data_is_read: bool = is_file_exist_loading and os.path.exists(cc_data_save_file)

    if cc_data_is_read:
        cc_data: AnnData = read_h5ad(cc_data_save_file)
    else:
        # cell-cell network
        cc_data = obtain_cell_cell_network(
            adata=adata,
            k=k,
            or_k=or_k,
            weight=weight,
            kernel=kernel,
            local_k=local_k,
            gamma=kernel_gamma,
            is_simple=is_simple
        )

    del adata

    smknn_time = cc_data.uns["elapsed_time"]

    if save_path is not None and not cc_data_is_read:
        save_h5ad(cc_data, file=cc_data_save_file)

    """
    5. Random walk with weighted seed cells
    """

    random_walk_is_read: bool = is_file_exist_loading and os.path.exists(
        random_walk_save_file) and is_save_random_walk_model

    if random_walk_is_read:
        random_walk: RandomWalk = read_pkl(random_walk_save_file)
    else:
        # random walk
        # noinspection DuplicatedCode
        random_walk: RandomWalk = RandomWalk(
            cc_adata=cc_data,
            init_status=init_score,
            epsilon=epsilon,
            max_steps=max_steps,
            gamma=gamma,
            enrichment_gamma=enrichment_gamma,
            p=p,
            n_jobs=n_jobs,
            min_seed_cell_rate=min_seed_cell_rate,
            max_seed_cell_rate=max_seed_cell_rate,
            credible_threshold=credible_threshold,
            enrichment_threshold=enrichment_threshold,
            is_ablation=is_ablation,
            is_simple=is_simple
        )

        if save_path is not None and random_walk_is_read:
            save_pkl(random_walk, save_file=random_walk_save_file)

    del random_walk_is_read, init_score, cc_data

    trs = _run_random_walk_(random_walk, is_ablation, is_simple)
    trs.uns["params"] = params

    del params

    random_walk_time = random_walk.elapsed_time

    # end time
    elapsed_time = time.perf_counter() - start_time
    step_time = poisson_vi_time + overlap_time + init_score_time + smknn_time + random_walk_time

    if elapsed_time < step_time:
        elapsed_time = step_time

    ul.log(__name__).info(f"Algorithm {project_name} consumes a total of {elapsed_time:.2f}s.")

    trs.uns["elapsed_time"] = {
        "PoissonVI": poisson_vi_time,
        "Overlap": overlap_time,
        "initial TRS": init_score_time,
        "SM-kNN": smknn_time,
        "Random walk": random_walk_time,
        "Total time": elapsed_time
    }

    if save_path is not None:
        save_h5ad(trs, file=trs_save_file)

    return trs


def association_score(
    adata: AnnData,
    score_name: str = "association_score",
    layer: str = "trs_source",
    axis: Literal[0, 1] = 0
) -> None:
    """
    Calculate association score for traits or diseases.
    This function calculates the association score for traits or diseases
    based on the TRS (Trait Relevance Score) data in the input AnnData object.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object containing TRS data.
    score_name : str, optional
        Name of the score column in the AnnData object. Default is "association_score".
    layer : str, optional
        Layer name in the AnnData object containing TRS data. Default is "trs_source".
    axis : Literal[0, 1], optional
        Axis to calculate the score (0 for traits, 1 for diseases). Default is 0.
    :return: None
    """
    if layer not in adata.layers:
        ul.log(__name__).error(f"Layer `{layer}` is not in the input AnnData object.")
        raise ValueError(f"Layer `{layer}` is not in the input AnnData object.")

    trs_source = to_dense(adata.layers[layer], is_array=True)

    trs_source_copy = trs_source.copy()
    trs_source_copy[trs_source_copy > 0] = 1

    trs_count_sum = trs_source_copy.sum(axis=axis)
    trs_count_sum[trs_count_sum == 0] = 1
    relevance_value = trs_source.sum(axis=axis) / trs_count_sum

    if axis == 0:
        adata.var[score_name] = np.array(relevance_value).flatten() * adata.shape[1]
        ul.log(__name__).info(f"View this related result `adata.var[{score_name}]`.")
    elif axis == 1:
        adata.obs[score_name] = np.array(relevance_value).flatten() * adata.shape[0]
        ul.log(__name__).info(f"View this related result `adata.obs[{score_name}]`.")
    else:
        ul.log(__name__).error("The `axis` parameter supports only 0 and 1.")
        raise ValueError("The `axis` parameter supports only 0 and 1.")

    sorted_indices = np.argsort(relevance_value)[::, ]
    ranks = np.zeros_like(relevance_value).tolist()

    for rank, index in enumerate(sorted_indices, start=1):
        ranks[index] = rank

    adata.var["rank"] = ranks
    adata.uns["association_score"] = {
        "layer": layer,
        "axis": axis,
        "causal_variant_sets_count": len(ranks)
    }


def knock(
    trs: AnnData,
    sc_atac: AnnData,
    da_peaks: AnnData,
    cc_data: AnnData,
    knock_trait: str,
    knock_info: dict[str, Union[str, collection]],
    knock_value: float = 0,
    is_add_control: bool = False
) -> AnnData:
    """
    Perform gene knockdown or knockout analysis on a specific trait.

    This function simulates the effect of knocking down or knocking out specific variants
    associated with a trait, and re-runs the random walk algorithm to compute the
    resulting TRS (Trait Relevance Score) changes.

    Parameters
    ----------
    trs : AnnData
        TRS result data from `ml.core`, containing parameters, variants, trait_info and trs_source.
    sc_atac : AnnData
        scATAC-seq data used in the original analysis.
    da_peaks : AnnData
        Differential accessibility peaks data from the original analysis.
    cc_data : AnnData
        Cell-cell similarity network data from the original analysis.
    knock_trait : str
        The trait ID to perform knockdown/knockout on.
    knock_info : dict[str, Union[str, collection]]
        Dictionary mapping knock group names to variant IDs (rsId) to be knocked down.
        Each key is a group name, and each value is either a single variant ID (str)
        or a collection of variant IDs to knock down together.
    knock_value : float, default 0
        The value to set for knocked-down variants. Default is 0 (complete knockout).
        Values >= 1e-3 are not recommended as they may not achieve the desired effect.
    is_add_control : bool, default False
        Whether to add control experiments (knocking out background variants).

    Returns
    -------
    AnnData
        AnnData object containing TRS results after knockdown/knockout.
        Includes knock parameters in .uns["params"] and knock-specific metadata.
    """
    if "params" not in trs.uns:
        ul.log(__name__).error(
            "`params` is not in `trs.uns`, please execute function `ml.core` first to obtain the result as input for the `trs` parameter.")
        raise ValueError(
            "`params` is not in `trs.uns`, please execute function `ml.core` first to obtain the result as input for the `trs` parameter.")

    if "variants" not in trs.uns:
        ul.log(__name__).error(
            "`variants` is not in `trs.uns`, please execute function `ml.core` first to obtain the result as input for the `trs` parameter.")
        raise ValueError(
            "`variants` is not in `trs.uns`, please execute function `ml.core` first to obtain the result as input for the `trs` parameter.")

    if "trait_info" not in trs.uns:
        ul.log(__name__).error(
            "`trait_info` is not in `trs.uns`, please execute function `ml.core` first to obtain the result as input for the `trs` parameter.")
        raise ValueError(
            "`trait_info` is not in `trs.uns`, please execute function `ml.core` first to obtain the result as input for the `trs` parameter.")

    if "trs_source" not in trs.layers:
        ul.log(__name__).error(
            "`trs_source` is not in `trs.layers`, please execute function `ml.core` first to obtain the result as input for the `trs` parameter.")
        raise ValueError(
            "`trs_source` is not in `trs.layers`, please execute function `ml.core` first to obtain the result as input for the `trs` parameter.")

    if knock_value >= 1E-3:
        ul.log(__name__).warning(
            "The value set for `knock_value` here is greater than 1e-3, which can easily fail to achieve the desired effect.")

    # param information
    params = trs.uns["params"]

    # variant information
    variants = trs.uns["variants"]
    trait_info = trs.uns["trait_info"]

    knock_variants: dict = {}
    knock_variants_bg: dict = {}

    if knock_trait not in trs.var["id"]:
        ul.log(__name__).error(f"`{knock_trait}` trait or disease does not exist.")
        raise ValueError(f"`{knock_trait}` trait or disease does not exist.")

    trait_obs = variants[knock_trait].obs
    knock_variant: AnnData = variants[knock_trait].copy()
    knock_variant_value = to_dense(knock_variant.X)

    _trait_info_ = trait_info[trait_info["id"] == knock_trait].values[0]
    knock_trait_info = pd.DataFrame(columns=trait_info.columns)

    ul.log(__name__).info(f"Knockdown or knockout settings for `{knock_trait}` trait or disease.")

    for k, v in knock_info.items():

        if isinstance(v, str):
            if v not in trait_obs["rsId"].tolist():
                ul.log(__name__).error(f"`{v}` variant does not exist in `{knock_trait}` trait or disease")
                raise ValueError(f"`{v}` variant does not exist in `{knock_trait}` trait or disease")
        elif isinstance(v, collection):
            if not set(v).issubset(set(trait_obs["rsId"].tolist())):
                ul.log(__name__).error(f"`{v}` variants does not exist in `{knock_trait}` trait or disease")
                raise ValueError(f"`{v}` variants does not exist in `{knock_trait}` trait or disease")

        need_k_index = trait_obs["rsId"].isin(list([v] if isinstance(v, str) else v)).values
        knock_variant_k = knock_variant.copy()
        knock_variant_k_value = knock_variant_value.copy()
        knock_variant_k_value[need_k_index, :] = knock_value
        knock_variant_k.X = to_sparse(knock_variant_k_value)
        knock_variants[k] = knock_variant_k

        knock_variant_k_bg = knock_variant.copy()
        knock_variant_k_value_bg = knock_variant_value.copy()
        knock_variant_k_value_bg[~need_k_index, :] = knock_value
        knock_variant_k_bg.X = to_sparse(knock_variant_k_value_bg)
        knock_variants_bg[k] = knock_variant_k_bg

        _trait_info_[0] = k
        knock_trait_info.loc[knock_trait_info.shape[0], :] = _trait_info_

    knock_trait_info.index = knock_trait_info["id"].astype(str)

    ul.log(__name__).info(f"Run the process after knocking down or knocking out.")
    knock_overlap_adata: AnnData = overlap_sum(sc_atac, knock_variants, knock_trait_info)

    # intermediate score data, integration data
    knock_init_score: AnnData = calculate_init_score_weight(
        adata=sc_atac,
        da_peaks_adata=da_peaks,
        overlap_adata=knock_overlap_adata,
        diff_peak_value=params["diff_peak_value"],
        is_simple=True,
        block_size=params["block_size"]
    )

    # random walk
    # noinspection DuplicatedCode
    knock_random_walk: RandomWalk = RandomWalk(
        cc_adata=cc_data,
        init_status=knock_init_score,
        epsilon=params["epsilon"],
        gamma=params["gamma"],
        enrichment_gamma=params["enrichment_gamma"],
        p=params["p"],
        n_jobs=params["n_jobs"] if "n_jobs" in params else -1,
        min_seed_cell_rate=params["min_seed_cell_rate"],
        max_seed_cell_rate=params["max_seed_cell_rate"],
        credible_threshold=params["credible_threshold"],
        enrichment_threshold=params["enrichment_threshold"],
        is_ablation=False,
        is_simple=True
    )

    # RUN
    knock_random_walk.run_knock(trs, knock_trait, False)

    if is_add_control:
        knock_random_walk.run_knock(trs, knock_trait, True)

    # Obtain result data
    knock_trs = knock_random_walk.trs_adata

    knock_trs.uns["params"] = {
        "knock_trait": knock_trait,
        "knock_info": knock_info,
        "knock_value": knock_value,
        "is_add_control": is_add_control,
        "run_core_params": params
    }

    trs.uns["is_knock"] = True

    if "pp_sum" in knock_trs.var.columns:
        knock_trs.var["pp_sum"] = knock_trs.var["pp_sum"].astype(float)

    if "pp_mean" in knock_trs.var.columns:
        knock_trs.var["pp_mean"] = knock_trs.var["pp_mean"].astype(float)

    if "count" in knock_trs.var.columns:
        knock_trs.var["count"] = knock_trs.var["count"].astype(int)

    # save result
    if params["save_path"] is not None:
        save_h5ad(knock_trs, file=os.path.join(params["save_path"], f"knock_trs_{knock_trait}.h5ad"))

    return knock_trs
