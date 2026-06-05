# -*- coding: UTF-8 -*-

import math
import time
from typing import Union, Tuple, Literal, Dict, Any, List

import torch
import torch.nn as nn

from torch import Tensor
from tqdm import tqdm
from joblib import Parallel, delayed

import numpy as np
from anndata import AnnData
from pandas import DataFrame

from . import mean_symmetric_scale, min_max_norm, perturb_data

from .. import util as ul
from ..util import (
    matrix_data,
    to_sparse,
    to_dense,
    collection,
    check_adata_get,
    enrichment_optional,
    check_gpu_availability,
    sparse_data
)

__name__: str = "tool_random_walk"


def _random_walk_cpu_(
    init_prob: matrix_data,
    weight: matrix_data,
    gamma: float = 0.05,
    epsilon: float = 1e-5,
    max_steps: int = 300,
    p: int = 2,
    is_perfect: bool = False
) -> collection:
    """
    Perform a random walk with restart on CPU.

    Parameters
    ----------
    init_prob : Union[list, np.ndarray, np.matrix]
        Initial probability distribution.
    weight : matrix_data, optional
        Transition probability matrix (weight matrix). Defaults to None.
    gamma : float, optional
        Restart probability (probability of jumping back to seed cells). Defaults to 0.05.
    epsilon : float, optional
        Convergence threshold for stopping criterion. Defaults to 1e-5.
    max_steps : int, optional
        Maximum number of iterations. Defaults to 300.
    p : int, optional
        Order of the norm used for convergence check {1: Manhattan, 2: Euclidean}. Defaults to 2.
    is_perfect : bool, optional
        If True, performs a "perfect" random walk where each sample (column) stops iterating 
        individually once it converges, rather than waiting for all samples to converge. 
        This ensures that converged samples retain their final values without further updates. 
        Defaults to False.

    Returns
    -------
    collection
        The stationary distribution after random walk convergence.
    """

    if isinstance(init_prob, collection):
        init_prob = np.asarray(init_prob, dtype=float)

    if len(init_prob.shape) == 1:
        v0: matrix_data = np.asarray(init_prob, dtype=float).ravel()[:, np.newaxis]
    else:
        row_number, col_number = init_prob.shape

        if row_number == 1 or col_number == 1:
            v0: matrix_data = np.asarray(init_prob, dtype=float).ravel()[:, np.newaxis]
        else:
            v0: matrix_data = init_prob

        del row_number, col_number

    element_number, sample_number = v0.shape

    if element_number != weight.shape[1]:
        ul.log(__name__).error(
            f"Dimension mismatch: initial probability distribution shape {v0.shape} is incompatible with "
            f"weight matrix shape {weight.shape}. Expected v0.shape[0] ({element_number}) to equal "
            f"weight.shape[1] ({weight.shape[1]})."
        )
        raise ValueError(
            f"Dimension mismatch: initial probability distribution shape {v0.shape} is incompatible with "
            f"weight matrix shape {weight.shape}. Expected v0.shape[0] ({element_number}) to equal "
            f"weight.shape[1] ({weight.shape[1]})."
        )

    del init_prob, element_number

    vt: matrix_data = v0.copy()
    step: int = 0
    delta: np.ndarray = np.ones(sample_number, dtype=np.float32)

    if is_perfect:
        vt_finish: matrix_data = np.zeros(v0.shape)
        label: np.ndarray = np.zeros(sample_number, dtype=int)

    del sample_number

    # iteration
    while np.any(delta > epsilon) and max_steps > step:

        if hasattr(weight, "dot"):
            v1 = (1 - gamma) * weight.dot(vt) + gamma * v0
        else:
            v1 = (1 - gamma) * np.dot(weight, vt) + gamma * v0

        # 1 and 2, It would be faster alone
        if p == 1:
            delta = np.abs(vt - v1).sum(axis=0)
        elif p == 2:
            delta = np.sqrt(np.square(np.abs(vt - v1)).sum(axis=0))
        else:
            delta = np.float_power(np.float_power(np.abs(vt - v1), p).sum(axis=0), 1.0 / p)

        vt = v1
        step += 1

        if is_perfect:
            is_label = delta > epsilon
            label[~is_label] += 1
            vt_finish[:, label == 1] = vt[:, label == 1]
            vt[:, label == 1] = 0

    if is_perfect:
        vt = vt_finish

    return vt.flatten() if vt.shape[1] == 1 else vt


class RandomWalkModel(nn.Module):
    """
    Random walk model for cell type association analysis.
    """

    def __init__(
        self,
        gamma: float = 0.05,
        epsilon: float = 1e-5,
        max_steps: int = 300,
        p: int = 2,
        device: str = 'auto',
        is_perfect: bool = False
    ):
        """
        Initialize the random walk model.

        Parameters
        ----------
        gamma : float, optional
            Restart probability (probability of jumping back to seed cells). Defaults to 0.05.
        epsilon : float, optional
            Convergence threshold for stopping criterion. Defaults to 1e-5.
        max_steps : int, optional
            Maximum number of iterations. Defaults to 300.
        p : int, optional
            Order of the norm used for convergence check {1: Manhattan, 2: Euclidean}. Defaults to 2.
        device : str, optional
            Device to run the model. Defaults to 'auto'.
        is_perfect : bool, optional
            If True, performs a "perfect" random walk where each sample (column) stops iterating
            individually once it converges, rather than waiting for all samples to converge.
            This ensures that converged samples retain their final values without further updates.
            Defaults to False.
        """
        super().__init__()
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_steps = max_steps
        self.p = p
        self.is_perfect = is_perfect

        is_gpu_available = check_gpu_availability()

        self.device = 'cuda' if device == 'auto' and is_gpu_available else device

        self.factor = 1 - self.gamma

    def forward(self, init_prob: Tensor, weight: Tensor) -> Tensor:
        """
        Perform a random walk with restart on GPU.

        Parameters
        ----------
        init_prob : Tensor
            Initial probability distribution.
        weight : Tensor
            Transition probability matrix (weight matrix). Defaults to None.

        Returns
        -------
        Tensor
            The stationary distribution after random walk convergence.
        """

        v0: Tensor = init_prob

        vt: Tensor = v0.clone()
        step: int = 0

        delta: Tensor = torch.ones(v0.shape[1], device=self.device, dtype=torch.float32).cuda()

        if self.is_perfect:
            vt_finish: Tensor = torch.zeros(v0.shape).cuda()
            label: Tensor = torch.ones(v0.shape[1], device=self.device, dtype=torch.int8).cuda()

        while (delta > self.epsilon).any() and self.max_steps > step:

            v1 = self.factor * torch.matmul(weight, vt) + self.gamma * v0

            delta = torch.linalg.norm(vt - v1, ord=self.p, dim=0)

            vt = v1
            step += 1

            if self.is_perfect:
                is_label = delta > self.epsilon
                label[~is_label] += 1
                vt_finish[:, label == 1] = vt[:, label == 1]
                vt[:, label == 1] = 0

        if self.is_perfect:
            vt = vt_finish

        return vt.flatten() if vt.shape[1] == 1 else vt


class WalkRandomDataParallel(nn.DataParallel):
    """
    Data parallel module for trait analysis.
    """

    def scatter(
        self,
        inputs: Tuple[torch.Tensor, torch.Tensor],
        kwargs: Dict[str, Any],
        device_ids: List[int]
    ) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[Dict[str, Any]]]:
        """
        Scatter the input data to multiple devices.

        Parameters
        ----------
        inputs : list
            List of input data to be scattered.
        kwargs : dict
            Dictionary of keyword arguments to be scattered.
        device_ids : list
            List of device IDs to scatter the data to.

        Returns
        -------
        tuple
            Tuple of scattered input data and keyword arguments.
        """
        _init_prob_, _weight_ = inputs
        scattered_init_prob = torch.nn.parallel.scatter(_init_prob_, device_ids, dim=1)
        scattered_weight = [_weight_.to(device) for device in device_ids]
        scattered_inputs = list(zip(scattered_init_prob, scattered_weight))

        # Handling kwargs (hyperparameters such as gamma, epsilon, etc.)
        scattered_kwargs = []

        for device_id in device_ids:
            device_kwargs = {}

            for key, value in kwargs.items():
                # Only Tensor type parameters require mobile devices
                if isinstance(value, torch.Tensor):
                    device_kwargs[key] = value.to(device_id)
                else:
                    device_kwargs[key] = value

            scattered_kwargs.append(device_kwargs)

        return scattered_inputs, scattered_kwargs

    def gather(self, outputs, output_device):
        """
        Collect the results after parallel processing, check for the existence of results,
        and merge the results by column (each result matrix has the same number of rows but different numbers of columns).

        Parameters
        ----------
        outputs : list
            Output results of each device
        output_device : int
            Output device ID.

        Returns
        -------
        Tensor
            The merged results sorted by column.
        """
        return torch.nn.parallel.scatter_gather.gather(outputs, output_device, dim=1)


def _random_walk_gpu_(
    init_prob: matrix_data,
    weight: matrix_data,
    gamma: float = 0.05,
    epsilon: float = 1e-5,
    max_steps: int = 300,
    p: int = 2,
    is_perfect: bool = False,
    device: str = 'auto'
) -> matrix_data:
    """
    Random walk analysis on GPU.

    Parameters
    ----------
    init_prob : matrix_data
        Initial probability distribution.
    weight : matrix_data
        Transition probability matrix (weight matrix). Defaults to None.
    gamma : float
        Random walk parameter. Defaults to 0.05.
    epsilon : float
        Convergence threshold. Defaults to 1e-5.
    max_steps : int
        Maximum number of steps. Defaults to 300.
    p : int
        Order of the random walk. Defaults to 2.
    is_perfect : bool, optional
        If True, performs a "perfect" random walk where each sample (column) stops iterating
        individually once it converges, rather than waiting for all samples to converge.
        This ensures that converged samples retain their final values without further updates.
        Defaults to False.
    device : str
        Device to run the analysis on. Defaults to 'auto'.

    Returns
    -------
    matrix_data
        The association score matrix, where each column represents the association score of a seed cell.
    """

    model = RandomWalkModel(gamma, epsilon, max_steps, p, device, is_perfect)

    device = model.device
    model.to(device)

    init_prob = torch.as_tensor(init_prob, device=device, dtype=torch.float32)

    if not isinstance(weight, torch.Tensor):

        if hasattr(weight, 'toarray'):
            weight = torch.sparse_coo_tensor(
                torch.as_tensor(np.array(weight.nonzero()), device=device),
                torch.as_tensor(weight.data, device=device, dtype=torch.float32),
                weight.shape,
                device=device,
                dtype=torch.float32
            )
        else:
            weight = torch.as_tensor(weight, device=device, dtype=torch.float32)

            if weight.dim() == 2:
                weight = weight.to_sparse()
    else:
        weight = weight.to(device)

        if not weight.is_sparse:
            weight = weight.to_sparse()

    if device == 'cuda' and 1 < torch.cuda.device_count() < init_prob.shape[1]:
        model = WalkRandomDataParallel(model)

    with torch.no_grad():
        result = model(init_prob, weight)

    return result.cpu().numpy()


def random_walk(
    seed_cell_weight: matrix_data,
    weight: matrix_data,
    gamma: float = 0.05,
    epsilon: float = 1e-5,
    max_steps: int = 300,
    p: int = 2,
    is_perfect: bool = False,
    device: str = 'auto'
) -> matrix_data:
    """
    Random walk analysis.

    Parameters
    ----------
    seed_cell_weight : matrix_data
        Seed cell weight matrix, where each column represents a seed cell.
    weight : matrix_data
        Transition probability matrix (weight matrix). Defaults to None.
    gamma : float
        Random walk parameter. Defaults to 0.05.
    epsilon : float
        Convergence threshold. Defaults to 1e-5.
    max_steps : int
        Maximum number of steps. Defaults to 300.
    p : int
        Order of the random walk. Defaults to 2.
    is_perfect : bool, optional
        If True, performs a "perfect" random walk where each sample (column) stops iterating
        individually once it converges, rather than waiting for all samples to converge.
        This ensures that converged samples retain their final values without further updates.
        Defaults to False.
    device : str
        Device to run the analysis on. Defaults to 'auto'.

    Returns
    -------
    matrix_data
        The association score matrix, where each column represents the association score of a seed cell.
    """

    availability = check_gpu_availability()

    if device == 'cpu' or (device == 'auto' and not availability):
        return _random_walk_cpu_(
            init_prob=seed_cell_weight,
            weight=weight,
            gamma=gamma,
            epsilon=epsilon,
            max_steps=max_steps,
            p=p,
            is_perfect=is_perfect
        )
    elif device == 'cuda' or (device == 'auto' and availability):

        try:
            return _random_walk_gpu_(
                init_prob=seed_cell_weight,
                weight=weight,
                gamma=gamma,
                epsilon=epsilon,
                max_steps=max_steps,
                p=p,
                is_perfect=is_perfect,
                device='cuda'
            )
        except RuntimeError as e:
            ul.log(__name__).warning(f"GPU failed to run, try to switch to CPU running.\n {e}")
            return _random_walk_cpu_(
                init_prob=seed_cell_weight,
                weight=weight,
                gamma=gamma,
                epsilon=epsilon,
                max_steps=max_steps,
                p=p,
                is_perfect=is_perfect
            )
    else:
        ul.log(__name__).error(
            f'The `device` ({device}) is not supported. Only supports "cpu", "cuda", and "auto" values.'
        )
        raise ValueError(f'The `device` ({device}) is not supported. Only supports "cpu", "cuda", and "auto" values.')


def scale_normalize(score: matrix_data, axis: Literal[0, 1, -1] = 0, is_verbose: bool = True) -> matrix_data:
    """
    Standardize and normalize the cell scores.

    Parameters
    ----------
    score : matrix_data
        Cell scores matrix.
    axis : Literal[0, 1, -1]
        Axis to apply the standardization and normalization. Defaults to 0.
    is_verbose : bool
        Whether to print the progress. Defaults to True.

    Returns
    -------
    matrix_data
        The standardized and normalized cell scores matrix.
    """
    cell_value = mean_symmetric_scale(score, axis=axis, is_verbose=is_verbose)
    cell_value = np.log1p(min_max_norm(cell_value, axis=axis))
    return cell_value


class RandomWalk:
    """
    Random walk analysis.
    """

    def __init__(
        self,
        cc_adata: AnnData,
        init_status: AnnData,
        epsilon: float = 1e-05,
        max_steps: int = 300,
        gamma: float = 0.05,
        enrichment_gamma: float = 0.05,
        p: int = 2,
        is_perfect: bool = False,
        n_jobs: int = -1,
        min_seed_cell_rate: float = 0.01,
        max_seed_cell_rate: float = 0.05,
        credible_threshold: float = 0,
        enrichment_threshold: Union[enrichment_optional, float] = 'golden',
        benchmark_count: int = 10,
        is_ablation: bool = False,
        is_simple: bool = True
    ):
        """
        Perform random walk steps.

        Parameters
        ----------
        cc_adata : AnnData
            Cell features.
        init_status : AnnData
            Cell scores under each trait.
        epsilon : float
            Convergence threshold for stopping criterion in random walk.
        max_steps : int
            Maximum number of steps in a random walk with restart.
        gamma : float
            Restart probability for random walk.
        enrichment_gamma : float
            Restart probability for random walk for enrichment.
        p : int
            Order of the norm used for loss {1: Manhattan distance, 2: Euclidean distance}.
        is_perfect : bool, optional
            If True, performs a "perfect" random walk where each sample (column) stops iterating
            individually once it converges, rather than waiting for all samples to converge.
            This ensures that converged samples retain their final values without further updates.
            Defaults to False.
        n_jobs : int
            The maximum number of concurrently running jobs.
        min_seed_cell_rate : float
            The minimum percentage of seed cells in all cells.
        max_seed_cell_rate : float
            The maximum percentage of seed cells in all cells.
        credible_threshold : float
            The threshold for determining the credibility of enriched cells in the context of
            enrichment, i.e. the threshold for judging enriched cells.
        enrichment_threshold : Union[str, float]
            Only by setting a threshold for the standardized output TRS can a portion of the
            enrichment results be obtained. Parameters support string types {'golden', 'half', 'e', 'pi', 'none'}, or
            valid floating-point types within the range of (0, log1p(1)).
        benchmark_count : int
            Number of benchmark runs.
        is_ablation : bool
            True represents obtaining the results of the ablation experiment. This parameter is limited
            by the `is_simple` parameter, and its effectiveness requires setting `is_simple` to `False`.
        is_simple : bool
            True represents not adding unnecessary intermediate variables, only adding the final result.
            It is worth noting that when set to `True`, the `is_ablation` parameter will become invalid, and when set
            to `False`, `is_ablation` will only take effect.

        Returns
        -------
        None
            Stable distribution score is stored in object attributes.
        """
        ul.log(__name__).info("Random walk with weighted seed cells.")

        start_time = time.perf_counter()

        # judge length
        if cc_adata.shape[0] != init_status.shape[0]:
            ul.log(__name__).error(
                f"The number of rows {cc_adata.shape[0]} in the data is not equal to the initialization state length "
                f"{np.array(init_status).size}"
            )
            raise ValueError(
                f"The number of rows {cc_adata.shape[0]} in the data is not equal to the initialization state length "
                f"{np.array(init_status).size}"
            )

        if p <= 0:
            ul.log(__name__).error(
                "The value of `p` must be greater than zero. Distance used for loss {1: Manhattan distance, "
                "2: Euclidean distance}"
            )
            raise ValueError(
                "The value of `p` must be greater than zero. Distance used for loss {1: Manhattan distance, "
                "2: Euclidean distance}"
            )
        elif p > 3:
            ul.log(__name__).warning("Suggested value for `p` is 1 or 2.")

        if epsilon > 0.1:
            ul.log(__name__).warning(
                f"Excessive value of parameter `epsilon`=({epsilon}) can lead to incorrect iteration and poor "
                f"enrichment effect."
            )
        elif epsilon <= 0:
            ul.log(__name__).error("The parameter of `epsilon` must be greater than zero.")
            raise ValueError("The parameter of `epsilon` must be greater than zero.")

        if max_steps <= 0:
            ul.log(__name__).error("The `max_steps` parameter must be a natural number greater than 0.")
            raise ValueError("The `max_steps` parameter must be a natural number greater than 0.")

        if "clusters" not in init_status.obs.columns:
            ul.log(__name__).error(
                "Unsupervised clustering information must be included in column `clusters` of `init_datus.obs`."
            )
            raise ValueError(
                "Unsupervised clustering information must be included in column `clusters` of `init_datus.obs`."
            )

        init_status.obs["clusters"] = init_status.obs["clusters"].astype(str)

        self.epsilon = epsilon
        self.max_steps = max_steps
        self.gamma = gamma
        self.enrichment_gamma = enrichment_gamma
        self.is_perfect = is_perfect
        self.p = p
        self.n_jobs = n_jobs
        self.min_seed_cell_rate = min_seed_cell_rate
        self.max_seed_cell_rate = max_seed_cell_rate
        self.credible_threshold = credible_threshold
        self.is_simple = is_simple
        self.is_ablation = is_ablation
        self.benchmark_count = benchmark_count
        self._enrichment_seed_cell_min_count_ = 3

        self.is_gpu_available = check_gpu_availability()

        if not is_simple and self.is_ablation:
            if "cell_mutual_knn" not in cc_adata.layers:
                ul.log(__name__).error("The ablation requires `cell_mutual_knn` to be in `cc_adata.layers`.")
                raise ValueError("The ablation requires `cell_mutual_knn` to be in `cc_adata.layers`.")

        if isinstance(enrichment_threshold, float):

            if enrichment_threshold <= 0 or enrichment_threshold >= np.log1p(1):
                ul.log(__name__).warning(
                    "The `enrichment_threshold` parameter is not set within the range of (0, log1p(1)), "
                    "this parameter will become invalid."
                )
                ul.log(__name__).warning(
                    "It is recommended to set the `enrichment_threshold` parameter to the 'golden' value."
                )

            self.enrichment_threshold = enrichment_threshold
        elif enrichment_threshold == "golden":
            golden_ratio = (1 + math.sqrt(5)) / 2
            self.enrichment_threshold = np.log1p(1) / (1 + 1 / golden_ratio)
        elif enrichment_threshold == "half":
            self.enrichment_threshold = np.log1p(1) / 2
        elif enrichment_threshold == "e":
            self.enrichment_threshold = np.log1p(1) / np.e
        elif enrichment_threshold == "pi":
            self.enrichment_threshold = np.log1p(1) / np.pi
        elif enrichment_threshold == "none":
            self.enrichment_threshold = np.log1p(1)
        else:
            raise ValueError(
                "Invalid enrichment settings. The string type in the `enrichment_threshold` parameter only supports "
                "the following parameter 'golden', 'half', 'e', 'pi',  Alternatively, input a floating-point type "
                "value within the range of (0, log1p(1))"
            )

        # Enrichment judgment
        self.is_run_core = False
        self.is_run_ablation_m_knn = False
        self.is_run_ablation_ncw = False
        self.is_run_ablation_nsw = False
        self.is_run_ablation_ncsw = False

        self.is_run_enrichment = False
        self.is_run_en_ablation_m_knn = False
        self.is_run_en_ablation_ncw = False
        self.is_run_en_ablation_nsw = False
        self.is_run_en_ablation_ncsw = False

        self.is_benchmark = False
        self.cluster_size_factor = {}

        self.cell_affinity = to_dense(cc_adata.layers["cell_affinity"])

        self.init_status: AnnData = init_status
        self.trait_info: list = list(init_status.var["id"])

        self.trs_adata: AnnData = AnnData(np.zeros(init_status.shape), obs=init_status.obs, var=init_status.var)
        self.trs_adata.uns = init_status.uns
        self.trs_adata.layers["init_trs"] = to_sparse(init_status.X)

        self.cell_anno = self.trs_adata.obs

        if not is_simple:
            for _layer_ in init_status.layers:
                self.trs_adata.layers[_layer_] = to_sparse(init_status.layers[_layer_])

        self.cell_size: int = self.trs_adata.shape[0]

        # trait
        self.trait_list: list = list(self.trs_adata.var_names)
        self.trait_range = range(len(self.trait_list))

        self.trs_source = np.zeros(init_status.shape)
        self.trs_source_positive = np.zeros(init_status.shape)
        self.trs_source_negative = np.zeros(init_status.shape)

        if not is_simple and self.is_ablation:
            self.trs_m_knn_source = np.zeros(init_status.shape)
            self.trs_ncw_source = np.zeros(init_status.shape)
            self.trs_nsw_source = np.zeros(init_status.shape)
            self.trs_ncsw_source = np.zeros(init_status.shape)
            self.random_seed_cell = np.zeros(init_status.shape)

        # Transition Probability Matrix
        self.weight = self._get_weight_(cc_adata.X)

        if not is_simple and self.is_ablation:
            self.weight_m_knn = self._get_weight_(cc_adata.layers["cell_mutual_knn"])

        del cc_adata

        self.cluster_types, self.init_seed_cell_size = self._get_cluster_info_()

        (
            self.seed_cell_count,
            self.seed_cell_threshold,
            self.seed_cell_weight_nsw,
            self.seed_cell_weight,
            self.seed_cell_index,
            self.seed_cell_weight_en_nsw,
            self.seed_cell_weight_en
        ) = self._get_seed_cell_()

        if not is_simple and self.is_ablation:
            init_status_no_weight = check_adata_get(init_status, "init_trs_ncw")
            (
                self.seed_cell_count_nw,
                self.seed_cell_threshold_nw,
                self.seed_cell_weight_ncsw,
                self.seed_cell_weight_ncw,
                _,
                self.seed_cell_weight_en_ncsw,
                self.seed_cell_weight_en_ncw
            ) = self._get_seed_cell_(init_data=init_status_no_weight, info="ablation")

        del self.cell_affinity
        del init_status

        self.elapsed_time = time.perf_counter() - start_time

    def _random_walk_(
        self,
        seed_cell_data: matrix_data,
        weight: matrix_data = None,
        gamma: float = 0.05,
        device: str = 'auto'
    ) -> matrix_data:
        """
        Perform a random walk.

        Parameters
        ----------
        seed_cell_data : matrix_data
            Seed cells data.
        weight : matrix_data, optional
            Weight matrix. Defaults to None.
        gamma : float, optional
            Restart probability. Defaults to None.
        device : str, optional
            Device to run on. Defaults to None.

        Returns
        -------
        matrix_data
            The value after random walk.
        """

        if weight is None:
            w = self.weight
        else:
            w = weight

        def _core_(_device_: str):
            return random_walk(
                seed_cell_data,
                weight=w,
                gamma=gamma,
                epsilon=self.epsilon,
                max_steps=self.max_steps,
                p=self.p,
                is_perfect=self.is_perfect,
                device=_device_
            )

        if not self.is_gpu_available:
            return _core_('cpu')

        try:
            _data_ = _core_(device)
        except Exception as e:
            ul.log(__name__).warning(f"GPU failed to run, try to switch to CPU running.\n {e}")
            _data_ = _core_('cpu')

        return _data_

    def _random_walk_core_(self, seed_cell_data: matrix_data, weight: matrix_data = None) -> matrix_data:
        """
        Random walk code function.

        Parameters
        ----------
        seed_cell_data : matrix_data
            Seed cells data.
        weight : matrix_data, optional
            Weight matrix. Defaults to None.

        Returns
        -------
        matrix_data
            The value after random walk.
        """
        return self._random_walk_(seed_cell_data, weight, self.gamma)

    @staticmethod
    def _get_weight_(cell_cell_matrix: matrix_data) -> sparse_data:
        """
        Obtain weights in random walk.

        Parameters
        ----------
        cell_cell_matrix : matrix_data
            Cell to cell connectivity matrix.

        Returns
        -------
        sparse_data
            The weights used in the iteration of random walk.
            Assign different weight matrices to seed cells.
        """
        ul.log(__name__).info("Obtain transition probability matrix.")
        data_weight = to_dense(cell_cell_matrix, is_array=True)
        cell_sum_weight = data_weight.sum(axis=1)[:, np.newaxis]
        cell_sum_weight[cell_sum_weight == 0] = 1
        return to_sparse(data_weight / cell_sum_weight)

    def _get_cell_weight_(self, seed_cell_size: int) -> matrix_data:
        """
        Obtain cell weights in random walk.

        Parameters
        ----------
        seed_cell_size : int
            Number of seed cells.

        Returns
        -------
        matrix_data
            Cell weights.
        """
        _cell_cell_knn_: matrix_data = self.cell_affinity.copy()

        # Obtain numerical values for constructing a k-neighbor network
        cell_cell_affinity_sort = np.sort(_cell_cell_knn_, axis=1)
        cell_cell_value = cell_cell_affinity_sort[:, -(seed_cell_size + 1)]
        del cell_cell_affinity_sort
        _cell_cell_knn_[self.cell_affinity < np.array(cell_cell_value).flatten()[:, np.newaxis]] = 0
        return _cell_cell_knn_

    def _get_seed_cell_size_(self, cell_size: int) -> int:
        """
        Obtain seed cell size in random walk.

        Parameters
        ----------
        cell_size : int
            Cell size.

        Returns
        -------
        int
            Seed cell size.
        """
        seed_cell_size: int = self.init_seed_cell_size if self.init_seed_cell_size < cell_size else cell_size

        # Control the number of seeds
        if (seed_cell_size / self.cell_size) < self.min_seed_cell_rate:
            seed_cell_size = np.ceil(self.min_seed_cell_rate * self.cell_size).astype(int)
        elif (seed_cell_size / self.cell_size) > self.max_seed_cell_rate:
            seed_cell_size = np.ceil(self.max_seed_cell_rate * self.cell_size).astype(int)

        if seed_cell_size == 0:
            seed_cell_size = 3
        elif seed_cell_size > cell_size:
            seed_cell_size = cell_size

        return seed_cell_size

    def _get_cluster_info_(self) -> Tuple[list, int]:
        """
        Obtain cluster information in random walk.

        Returns
        -------
        Tuple[list, int]
            Cluster types, seed cell size.
        """
        # cluster size/count
        cluster_types = list(set(self.trs_adata.obs["clusters"]))
        cluster_types.sort()

        clusters = list(self.trs_adata.obs["clusters"])

        for cluster in cluster_types:
            count = clusters.count(cluster)
            self.cluster_size_factor.update({str(cluster): count})

        seed_cell_size = min(self.cluster_size_factor.values())

        self.trs_adata.uns["cluster_info"] = {
            "cluster_size_factor": self.cluster_size_factor,
            "min_seed_cell_rate": self.min_seed_cell_rate,
            "max_seed_cell_rate": self.max_seed_cell_rate,
            "init_seed_cell_size": seed_cell_size
        }
        return cluster_types, seed_cell_size

    def _get_seed_cell_clustering_weight_(self, seed_cell_index: collection) -> Tuple[collection, dict]:
        """
        This function is used to obtain the percentage of seed cells that occupy this cell type, i.e., the seed cell
        clustering weight. The purpose of this weight is to provide fair enrichment opportunities for those with fewer
        cell numbers in cell clustering types.

        Parameters
        ----------
        seed_cell_index : collection
            Index of seed cells.

        Returns
        -------
        Tuple[collection, dict]
            The seed cell clustering weight, equity factor.
        """
        cell_anno: DataFrame = self.cell_anno.copy()
        cell_clusters = cell_anno["clusters"].values
        seed_cell_cell_anno: DataFrame = cell_anno.iloc[seed_cell_index]

        seed_cell_cell_clusters: list = cell_clusters[seed_cell_index].tolist()
        seed_cell_cell_cluster_rate = {}

        for _k_, _v_ in self.cluster_size_factor.items():
            seed_cell_cell_cluster_rate.update({_k_: seed_cell_cell_clusters.count(_k_) / _v_})

        seed_cell_cluster_weight = seed_cell_cell_anno["clusters"].map(seed_cell_cell_cluster_rate).values
        return mean_symmetric_scale(seed_cell_cluster_weight, is_verbose=False), seed_cell_cell_cluster_rate

    def _get_seed_cell_weight_(
        self,
        seed_cell_index: collection,
        value: collection,
        seed_cell_index_enrichment: collection = None
    ) -> collection:
        """
        Obtain seed cell weight in random walk.

        Parameters
        ----------
        seed_cell_index : collection
            Index of seed cells.
        value : collection
            Seed cell values.
        seed_cell_index_enrichment : collection, optional
            Index of seed cells for enrichment. Defaults to None.

        Returns
        -------
        collection
            Seed cell weight.
        """

        if seed_cell_index_enrichment is None:
            seed_cell_index_enrichment = seed_cell_index

        # Calculate the degree of seed cells in the seed cell network
        seed_cell_mutual_knn = np.array(self.cell_affinity[seed_cell_index, :][:, seed_cell_index])
        seed_weight_degree: collection = seed_cell_mutual_knn.sum(axis=0)
        seed_weight_degree_weight = mean_symmetric_scale(seed_weight_degree, is_verbose=False)

        # Calculate the initialization score weight
        seed_cell_value = value[seed_cell_index_enrichment]
        seed_cell_value_weight = mean_symmetric_scale(seed_cell_value, is_verbose=False)

        # Percentage weight of seed cells in cell type clustering
        seed_cell_clustering_weight = self._get_seed_cell_clustering_weight_(seed_cell_index)[0]
        seed_weight_value = seed_weight_degree_weight * seed_cell_value_weight * seed_cell_clustering_weight

        # Calculate weight
        seed_weight_value = seed_weight_value / (1 if np.sum(seed_weight_value) == 0 else np.sum(seed_weight_value))
        return seed_weight_value

    def _get_seed_cell_(
        self,
        init_data: AnnData = None,
        info: str = None
    ) -> Tuple[collection, collection, matrix_data, matrix_data, matrix_data, matrix_data, matrix_data]:
        """
        Obtain information related to seed cells.

        Parameters
        ----------
        init_data : AnnData, optional
            Initial TRS data. Defaults to None.
        info : str, optional
            Log information about seed cells. Defaults to None.

        Returns
        -------
        Tuple[collection, collection, matrix_data, matrix_data, matrix_data, matrix_data, matrix_data]
            1. Set seed cell thresholds for each trait or disease.
            2. Seed cell weights obtained for each trait or disease based on the `init_data` parameter, with each seed
                cell assigned the same weight. Note that this only takes effect when `is_simple` is true.
            3. Seed cell weights obtained for each trait or disease based on the init_data parameter, and the weight of
                each seed cell will be assigned based on the similarity between cells.
            4. Seed cell index, which will be used for later knockout or knockdown prediction.
            5. Based on the init_data parameter, a reference seed cell weight is obtained for enrichment analysis
                assistance for each trait or disease, and each seed cell is assigned the same weight.
                Note that this only takes effect when `is_simple` is true.
            6. Reference seed cell weights for auxiliary enrichment analysis of each trait or disease based on the
                init_data parameter, and the weight of each seed cell will be assigned based on the similarity
                between cells.
        """

        if init_data is None:
            init_data = self.init_status

        n_traits = len(self.trait_list)
        n_cells = self.cell_size

        seed_cell_count = np.zeros(n_traits, dtype=int)
        seed_cell_threshold = np.zeros(n_traits)
        seed_cell_weight = np.zeros((n_cells, n_traits))
        seed_cell_index = np.zeros((n_cells, n_traits), dtype=int)
        seed_cell_weight_en = np.zeros((n_cells, n_traits))

        if not self.is_simple:
            seed_cell_matrix = np.zeros((n_cells, n_traits))
            seed_cell_matrix_en = np.zeros((n_cells, n_traits))
        else:
            seed_cell_matrix = np.zeros((1, 1))
            seed_cell_matrix_en = np.zeros((1, 1))

        ul.log(__name__).info(
            f"Calculate {n_traits} traits/diseases for seed cells information.{f' ({info})' if info else ''}"
        )

        trait_values_all = to_dense(init_data.X, is_array=True)

        def _process_single_trait(i: int) -> None:
            trait_value = trait_values_all[:, i]
            trait_value_max = trait_value.max()
            trait_value_min = trait_value.min()

            if trait_value_min == trait_value_max:
                return

            # Directly obtain descending index
            trait_value_sort_index = np.argsort(trait_value).astype(int)
            trait_value_sort_index = trait_value_sort_index[::-1]

            # Calculate the number of cells with>0
            _gt0_cell_size = (trait_value > 0).sum()

            _seed_cell_size = self._get_seed_cell_size_(_gt0_cell_size)

            seed_cell_count[i] = _seed_cell_size
            seed_cell_threshold[i] = trait_value[trait_value_sort_index[_seed_cell_size]]

            # Set seed cell index and weight
            _seed_cell_index = trait_value_sort_index[:_seed_cell_size]
            seed_cell_index[_seed_cell_index, i] = 1
            seed_cell_weight[_seed_cell_index, i] = self._get_seed_cell_weight_(
                seed_cell_index=_seed_cell_index, value=trait_value
            )

            # Enrichment interval index
            _enrichment_start = _seed_cell_size
            _enrichment_end = min(2 * _seed_cell_size, self.cell_size - 1)

            if _gt0_cell_size == _seed_cell_size:
                _enrichment_start = max(_seed_cell_size - self._enrichment_seed_cell_min_count_, 0)
                _enrichment_end = _seed_cell_size

            _seed_cell_en_index = trait_value_sort_index[_enrichment_start:_enrichment_end]
            seed_cell_weight_en[_seed_cell_en_index, i] = self._get_seed_cell_weight_(
                seed_cell_index=_seed_cell_index if len(_seed_cell_en_index) == len(_seed_cell_index) else _seed_cell_en_index,
                value=trait_value,
                seed_cell_index_enrichment=_seed_cell_en_index
            )

            if not self.is_simple and self.is_ablation:
                seed_cell_value = np.zeros(n_cells)
                seed_cell_value[_seed_cell_index] = 1

                if seed_cell_value.sum() == 0:
                    seed_cell_matrix[:, i] = seed_cell_value
                else:
                    seed_cell_matrix[:, i] = seed_cell_value / seed_cell_value.sum()

                seed_cell_en_value = np.zeros(n_cells)
                seed_cell_en_value[_seed_cell_en_index] = 1

                if seed_cell_en_value.sum() == 0:
                    seed_cell_matrix_en[:, i] = seed_cell_en_value
                else:
                    seed_cell_matrix_en[:, i] = seed_cell_en_value / seed_cell_en_value.sum()

        # Parallel processing of all traits and real-time display of progress
        Parallel(n_jobs=self.n_jobs, backend='threading')(
            delayed(_process_single_trait)(i) for i in
            tqdm(self.trait_range, desc="Obtain progress of seed cells with weights")
        )

        return seed_cell_count, seed_cell_threshold, seed_cell_matrix, seed_cell_weight, seed_cell_index, seed_cell_matrix_en, seed_cell_weight_en

    @staticmethod
    def scale_norm(score: matrix_data, is_verbose: bool = False) -> matrix_data:
        """
        Scale normalization of the score matrix.

        Parameters
        ----------
        score : matrix_data
            Score matrix.
        is_verbose : bool, optional
            Whether to print the progress. Defaults to False.

        Returns
        -------
        matrix_data
            The normalized score matrix.
        """
        return scale_normalize(score, axis=0, is_verbose=is_verbose)

    def _simple_error_(self) -> None:
        """
        Check if the parameter `is_simple` is True.

        Raises
        ------
        ValueError
        RuntimeError
        """

        if self.is_simple and "is_simple" in self.trs_adata.uns.keys() and self.trs_adata.uns["is_simple"]:
            ul.log(__name__).error("The parameter `is_simple` is True, so running this method is not supported.")
            raise RuntimeError("The parameter `is_simple` is True, so running this method is not supported.")

    def run_benchmark(self) -> None:
        """
        Perform random walk of random seeds on all traits.
        """
        self._simple_error_()

        ul.log(__name__).info(f"Calculate {len(self.trait_list)} traits/diseases for process `run_benchmark` \
                                (Count: {self.benchmark_count}). (Randomly perturb seed cells. ===> `benchmark`)")

        total_steps = len(self.trait_list) * self.benchmark_count
        with tqdm(total=total_steps) as pbar:
            for i in self.trait_range:

                random_seed_cell_matrix = np.zeros((self.cell_size, self.benchmark_count))

                # Obtain all cell score values in a trait
                trait_adata: AnnData = self.init_status[:, i]
                trait_value: collection = to_dense(trait_adata.X, is_array=True).flatten()

                # Obtain the maximum initial score
                trait_value_max = np.max(trait_value)
                trait_value_min = np.min(trait_value)

                for j in range(self.benchmark_count):
                    # Set random seed information
                    random_seed_cell = np.zeros(self.cell_size)
                    random_seed_index = np.random.choice(
                        np.arange(0, self.cell_size), size=self.seed_cell_count[i], replace=False
                    )

                    if trait_value_min != trait_value_max:
                        # seed cell weight
                        random_seed_cell[random_seed_index] = 1 / self.cell_size
                        random_seed_cell_matrix[:, j] = random_seed_cell

                    pbar.update(1)

                # Random walk
                cell_value_matrix = self._random_walk_core_(random_seed_cell_matrix)
                # Remove the influence of background
                self.random_seed_cell[:, i] = cell_value_matrix.mean(axis=1)

        cell_value = self.scale_norm(self.random_seed_cell)
        self.trs_adata.layers["benchmark"] = to_sparse(cell_value)
        self.is_benchmark = True

    @staticmethod
    def _get_label_description_(label: str) -> Tuple[str, str]:
        """
        Get the description of the label.

        Parameters
        ----------
        label : str
            Label of the information.

        Returns
        -------
        Tuple[str, str]
            Description of the label and the layer of the information.
        """
        if label == "run_core" or label == "run_en":
            return "Calculate random walk with weighted seed cells.", "trs"
        elif label == "run_ablation_ncsw" or label == "run_en_ablation_ncsw":
            return "Removed cell weights in random walk and cluster type weights in initial scores.", "trs_ncsw"
        elif label == "run_ablation_nsw" or label == "run_en_ablation_nsw":
            return "Removed cell weights from random walk.", "trs_nsw"
        elif label == "run_ablation_ncw" or label == "run_en_ablation_ncw":
            return "Removed cell cluster type weights in initial scores.", "trs_ncw"
        elif label == "run_ablation_m_knn" or label == "run_en_ablation_m_knn":
            return "Using the M-KNN method during the execution of weighted random walks.", "trs_m_knn"
        elif label == "run_knock (positive)":
            return "Run knockout or knockdown by random walk with weight. (positive)", "knock_effect_positive"
        elif label == "run_knock (negative)":
            return "Run knockout or knockdown by random walk with weight. (negative)", "knock_effect_negative"
        elif label == "run_knock_control (control & positive)":
            return "Run knockout or knockdown by random walk with weight. (control & positive)", "knock_effect_positive_control"
        elif label == "run_knock_control (control & negative)":
            return "Run knockout or knockdown by random walk with weight. (control & negative)", "knock_effect_negative_control"
        else:
            raise ValueError(f"{label} is not a valid information.")

    def _run_(self, seed_cell_data: matrix_data, label: str, weight: matrix_data = None) -> matrix_data:
        """
        Calculate random walk.

        Parameters
        ----------
        seed_cell_data : matrix_data
            Seed cell data.
        label : str
            Label of the information.
        weight : matrix_data, optional
            Weight matrix. Defaults to None.

        Returns
        -------
        matrix_data
            Seed cell data.
        """

        if weight is None:
            weight = self.weight

        _log_info_, _layer_label_ = self._get_label_description_(label)
        log_info_str: str = f"Calculate {len(self.trait_list)} traits/diseases for process `{label}`. ({_log_info_} ===> `{_layer_label_}`)"
        ul.log(__name__).info(log_info_str)

        score = self._random_walk_core_(seed_cell_data, weight=weight)

        ul.log(__name__).info("Normalize the results")
        cell_value = self.scale_norm(score)

        if _layer_label_ == "trs":
            self.trs_adata.X = to_sparse(cell_value)
        else:
            self.trs_adata.layers[_layer_label_] = to_sparse(cell_value)

        return score

    def run_core(self) -> None:
        """
        Calculate weighted random walk.
        """
        if not self.is_simple:
            self.trs_adata.layers["seed_cell_weight"] = to_sparse(self.seed_cell_weight)

        self.trs_adata.layers["seed_cell_index"] = to_sparse(self.seed_cell_index)

        self.trs_adata.var["seed_cell_count"] = self.seed_cell_count
        self.trs_adata.var["seed_cell_threshold"] = self.seed_cell_threshold
        self.trs_source = self._run_(self.seed_cell_weight, "run_core")

        self.trs_adata.layers["trs_source"] = to_sparse(self.trs_source)
        self.is_run_core = True

    def run_ablation_m_knn(self) -> None:
        """
        Using M-KNN fully connected cellular network.
        """
        self._simple_error_()
        self.trs_m_knn_source = self._run_(self.seed_cell_weight, "run_ablation_m_knn", self.weight_m_knn)
        self.is_run_ablation_m_knn = True

    def run_ablation_ncw(self) -> None:
        """
        Removed cell cluster type weights in initial scores
        """
        self._simple_error_()
        self.trs_adata.layers["seed_cell_weight_ncw"] = self.seed_cell_weight_ncw

        if "seed_cell_count_nw" not in self.trs_adata.var.columns:
            self.trs_adata.var["seed_cell_count_nw"] = self.seed_cell_count_nw

        if "seed_cell_threshold_nw" not in self.trs_adata.var.columns:
            self.trs_adata.var["seed_cell_threshold_nw"] = self.seed_cell_threshold_nw

        self.trs_ncw_source = self._run_(self.seed_cell_weight_ncw, "run_ablation_ncw")
        self.is_run_ablation_ncw = True

    def run_ablation_nsw(self) -> None:
        """
        Removed cell weights from random walk.
        """
        self._simple_error_()
        self.trs_adata.layers["seed_cell_weight_nsw"] = self.seed_cell_weight_nsw
        self.trs_nsw_source = self._run_(self.seed_cell_weight_nsw, "run_ablation_nsw")
        self.is_run_ablation_nsw = True

    def run_ablation_ncsw(self) -> None:
        """
        Removed cell weights in random walk and cluster type weights in initial scores.
        """
        self._simple_error_()
        self.trs_adata.layers["seed_cell_weight_ncsw"] = self.seed_cell_weight_ncsw

        if "seed_cell_count_nw" not in self.trs_adata.var.columns:
            self.trs_adata.var["seed_cell_count_nw"] = self.seed_cell_count_nw

        if "seed_cell_threshold_nw" not in self.trs_adata.var.columns:
            self.trs_adata.var["seed_cell_threshold_nw"] = self.seed_cell_threshold_nw

        self.trs_ncsw_source = self._run_(self.seed_cell_weight_ncsw, "run_ablation_ncsw")
        self.is_run_ablation_ncsw = True

    def _run_enrichment_(self, seed_cell_en_weight: matrix_data, label: str) -> None:
        """
        Enrichment analysis of traits/cells.

        Parameters
        ----------
        seed_cell_en_weight : matrix_data
            Seed cell data.
        label : str
            Label of the information.
        """

        _layer_label_: str = "tre"

        source_value: matrix_data = self.trs_source

        _, _trs_layer_label_ = self._get_label_description_(label)

        def _log_info_str_(_name_: str) -> str:
            return f"Need to run the `{_name_}` method first in order to run this method. Start run..."

        if label == "run_en":
            if not self.is_run_core:
                ul.log(__name__).warning(_log_info_str_("run_core"))
                self.run_core()

        elif label == "run_en_ablation_m_knn":
            if not self.is_run_ablation_m_knn:
                ul.log(__name__).warning(_log_info_str_("run_ablation_m_knn"))
                self.run_ablation_m_knn()

            _layer_label_ = "tre_m_knn"
            source_value = self.trs_m_knn_source

        elif label == "run_en_ablation_ncw":
            if not self.is_run_ablation_ncw:
                ul.log(__name__).warning(_log_info_str_("run_ablation_ncw"))
                self.run_ablation_ncw()

            _layer_label_ = "tre_ncw"
            source_value = self.trs_ncw_source

        elif label == "run_en_ablation_nsw":
            if not self.is_run_ablation_nsw:
                ul.log(__name__).warning(_log_info_str_("run_ablation_nsw"))
                self.run_ablation_nsw()

            _layer_label_ = "tre_nsw"
            source_value = self.trs_nsw_source

        elif label == "run_en_ablation_ncsw":
            if not self.is_run_ablation_ncsw:
                ul.log(__name__).warning(_log_info_str_("run_ablation_ncsw"))
                self.run_ablation_ncsw()

            _layer_label_ = "tre_ncsw"
            source_value = self.trs_ncsw_source

        else:
            log_info_str: str = f"{label} error. `run_en`, `run_en_ablation_m_knn`, `run_en_ablation_ncw`, `run_en_ablation_nsw` or `run_en_ablation_ncsw`"
            raise ValueError(log_info_str)

        cell_anno: DataFrame = self.cell_anno.copy()

        if label == "run_en":
            trs_score = to_dense(self.trs_adata.X, is_array=True)
        else:
            trs_score = to_dense(self.trs_adata.layers[_trs_layer_label_], is_array=True)

        # Initialize enriched container
        trait_cell_enrichment = np.zeros(self.trs_adata.shape).astype(int)
        trait_cell_credible = np.zeros(self.trs_adata.shape).astype(np.float32)

        ul.log(__name__).info(
            f"Calculate {len(self.trait_list)} traits/diseases for process `{label}`. (Enrichment-random walk)"
        )
        # Random walk
        cell_value_data = self._random_walk_(
            seed_cell_en_weight,
            weight=self.weight_m_knn if label == "run_en_ablation_m_knn" else self.weight,
            gamma=self.enrichment_gamma
        )

        ul.log(__name__).info(
            f"Calculate {len(self.trait_list)} traits/diseases for process `{label}`. (Enrichment-score)"
        )

        # Process each trait in parallel
        def _process_trait(i):
            # Random walk
            cell_value = cell_value_data[:, i]

            # separate
            cell_value_credible = mean_symmetric_scale(
                np.array(source_value[:, i]).flatten() - np.array(cell_value).flatten(), is_verbose=False)

            # This step is only executed if it contains cell clustering type weights
            if label == "run_en" or label == "run_en_ablation_nsw" or label == "run_en_ablation_m_knn":
                _enrichment_index_ = trs_score[:, i] > self.enrichment_threshold

                if np.any(_enrichment_index_):
                    # Ratio of cell clustering types enriched by threshold
                    _, _clustering_map_ = self._get_seed_cell_clustering_weight_(_enrichment_index_)
                    _clustering_weight_ = cell_anno["clusters"].map(_clustering_map_)
                    _clustering_weight_ = mean_symmetric_scale(_clustering_weight_, is_verbose=False)
                    _clustering_weight_mean_ = _clustering_weight_.mean()
                    # Correction score
                    cell_value_credible += (_clustering_weight_ - _clustering_weight_mean_)

            trait_cell_enrichment[:, i][cell_value_credible > self.credible_threshold] = 1
            trait_cell_credible[:, i] = cell_value_credible

        # Process each trait in parallel, backend='threading' can effectively prevent the read-only parameter issue caused by copying in loky multi-process mode
        Parallel(n_jobs=self.n_jobs, backend='threading')(delayed(_process_trait)(i) for i in tqdm(self.trait_range))

        self.trs_adata.layers[_layer_label_] = to_sparse(trait_cell_enrichment)

        if not self.is_simple:
            self.trs_adata.layers[f"credible_{_layer_label_}"] = to_sparse(trait_cell_credible)

    def run_enrichment(self) -> None:
        """
        Enrichment analysis.
        """
        self._run_enrichment_(self.seed_cell_weight_en, "run_en")
        self.is_run_enrichment = True

    def run_en_ablation_m_knn(self) -> None:
        """
        Using M-KNN fully connected cellular network (Enrichment analysis)
        """
        self._simple_error_()
        self._run_enrichment_(self.seed_cell_weight_en, "run_en_ablation_m_knn")
        self.is_run_en_ablation_m_knn = True

    def run_en_ablation_ncw(self) -> None:
        """
        Removed cell cluster type weights in initial scores.
        """
        self._simple_error_()
        self._run_enrichment_(self.seed_cell_weight_en_ncw, "run_en_ablation_ncw")
        self.is_run_en_ablation_ncw = True

    def run_en_ablation_nsw(self) -> None:
        """
        Removed cell weights from random walk
        """
        self._simple_error_()
        self._run_enrichment_(self.seed_cell_weight_en_nsw, "run_en_ablation_nsw")
        self.is_run_en_ablation_nsw = True

    def run_en_ablation_ncsw(self) -> None:
        """
        Removed cell weights in random walk and cluster type weights in initial scores.
        """
        self._simple_error_()
        self._run_enrichment_(self.seed_cell_weight_en_ncsw, "run_en_ablation_ncsw")
        self.is_run_en_ablation_ncsw = True

    def run_knock(self, trs: AnnData, knock_trait: str, is_control: bool = False) -> None:
        """
        Knockout analysis.

        Parameters
        ----------
        trs : AnnData
            Input `AnnData` object.
        knock_trait : str
            Knockout trait or disease.
        is_control : bool, optional
            Whether to control the knockout.
            default is False.
        """

        if trs.shape[0] != self.cell_size:
            ul.log(__name__).error(
                f"The number of cells ({trs.shape[0]}) in the input `trs` is inconsistent with the number of cells \
                 ({self.cell_size}) in the knockdown after knockout"
            )
            raise ValueError(
                f"The number of cells ({trs.shape[0]}) in the input `trs` is inconsistent with the number of cells \
                 ({self.cell_size}) in the knockdown after knockout."
            )

        if "trs_source" not in trs.layers:
            ul.log(__name__).error(
                "`trs_source` is not in `trs.layers`, please execute function `ml.core` first to obtain the result \
                 as input for the `trs` parameter."
            )
            raise ValueError(
                "`trs_source` is not in `trs.layers`, please execute function `ml.core` first to obtain the result \
                 as input for the `trs` parameter."
            )

        if "seed_cell_index" not in trs.layers:
            ul.log(__name__).error(
                "`seed_cell_index` is not in `trs.layers`, please execute function `ml.core` first to obtain the \
                 result as input for the `trs` parameter."
            )
            raise ValueError(
                "`seed_cell_index` is not in `trs.layers`, please execute function `ml.core` first to obtain the \
                 result as input for the `trs` parameter."
            )

        if knock_trait not in trs.var["id"]:
            ul.log(__name__).error(f"`{knock_trait}` trait or disease does not exist.")
            raise ValueError(f"`{knock_trait}` trait or disease does not exist.")

        knock_info_content = "run_knock_control" if is_control else "run_knock"

        ul.log(__name__).info(f"Calculate {len(self.trait_list)} for seed cells information. ({knock_info_content})")
        init_trait_source_value: matrix_data = to_dense(trs[:, knock_trait].layers["init_trs"])
        init_trait_value: matrix_data = to_dense(self.trs_adata.layers["init_trs"])

        init_trait_positive_effect = init_trait_source_value - init_trait_value
        init_trait_positive_effect[init_trait_positive_effect < 0] = 0
        init_trait_negative_effect = init_trait_value - init_trait_source_value
        init_trait_negative_effect[init_trait_negative_effect < 0] = 0

        self.trs_adata.layers["init_trait_positive"] = to_sparse(init_trait_positive_effect)
        self.trs_adata.layers["init_trait_negative"] = to_sparse(init_trait_negative_effect)

        init_trait_positive_adata = check_adata_get(self.trs_adata, "init_trait_positive")
        (_, positive_seed_cell_threshold, _, positive_seed_cell_weight, _, _, _,) = self._get_seed_cell_(
            init_data=init_trait_positive_adata, info="knock (positive)")

        init_trait_negative_adata = check_adata_get(self.trs_adata, "init_trait_negative")
        (_, negative_seed_cell_threshold, _, negative_seed_cell_weight, _, _, _,) = self._get_seed_cell_(
            init_data=init_trait_negative_adata, info="knock (negative)")

        self.trs_adata.var["positive_seed_cell_threshold"] = positive_seed_cell_threshold
        self.trs_adata.var["negative_seed_cell_threshold"] = negative_seed_cell_threshold

        if is_control:
            ul.log(__name__).info("Perturb the initialization TRS.")
            for i in tqdm(self.trait_range):
                positive_seed_cell_weight[:, i] = perturb_data(positive_seed_cell_weight[:, i], 1.0)
                negative_seed_cell_weight[:, i] = perturb_data(negative_seed_cell_weight[:, i], 1.0)

        # Obtain the result after random walk
        _positive_label_: str = "control & positive" if is_control else "positive"
        self.trs_source_positive = self._run_(
            positive_seed_cell_weight,
            f"{knock_info_content} ({_positive_label_})"
        )
        _negative_label_: str = "control & negative" if is_control else "negative"
        self.trs_source_negative = self._run_(
            negative_seed_cell_weight,
            f"{knock_info_content} ({_negative_label_})"
        )

        self.trs_adata.layers["knock_effect_positive_control" if is_control else "knock_effect_positive"] = to_sparse(
            self.trs_source_positive
        )
        self.trs_adata.layers["knock_effect_negative_control" if is_control else "knock_effect_negative"] = to_sparse(
            self.trs_source_negative
        )

        ul.log(__name__).info(
            "Obtain the effect size of knocking out or knocking down ==> .layers[\"{}\"]".format(
                "knock_effect_control" if is_control else "knock_effect"
            )
        )
        knock_effect_value = self.trs_source_positive - self.trs_source_negative
        self.trs_adata.layers["knock_effect_source_control" if is_control else "knock_effect_source"] = to_sparse(
            knock_effect_value
        )
        self.trs_adata.layers["knock_effect_control" if is_control else "knock_effect"] = to_sparse(
            mean_symmetric_scale(knock_effect_value, axis=0, is_verbose=False)
        )
