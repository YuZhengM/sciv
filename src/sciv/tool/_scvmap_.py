# -*- coding: UTF-8 -*-

from typing import Literal

import pandas as pd
import requests
from pandas import DataFrame
from requests import Response

from .. import util as ul

from ..util import scvmap_url

__name__: str = "tool_scvmap"

log = ul.log(__name__)

_Method = Literal['finemap', 'susie']
_Genome = Literal['hg38', 'hg19']
_Gene_set = Literal[
    'GO_Biological_Process_2023', 'GO_Cellular_Component_2023', 'GO_Molecular_Function_2023',
    'GWAS_Catalog_2023', 'KEGG_2016'
]


def get_result_data(resp: Response):
    json_data = resp.json()

    if json_data["status"]:
        return json_data["data"]

    raise ValueError(json_data["message"])


def request_get_data(path: str, **kwargs):
    log.info(f"Get request {scvmap_url}/{path}")
    response = requests.get(f"{scvmap_url}/{path}", **kwargs)
    return get_result_data(response)


def request_post_data(path: str, json: dict = None, **kwargs):
    log.info(f"Post request {scvmap_url}/{path}")

    if json:
        log.info(f"Post request parameters: {json}")

    response = requests.post(f"{scvmap_url}/{path}", json=json, **kwargs)
    return get_result_data(response)


def list_trait_info_data(trait_id: str, genome: _Genome = "hg38", method: _Method = "finemap") -> DataFrame:
    total_size = request_post_data(
        f"detail/trait_info/{trait_id}/{genome}/{method}",
        {
            "page": 1,
            "size": 1,
            "order": 0
        }
    )["total"]
    request_variant_info = request_post_data(
        f"detail/trait_info/{trait_id}/{genome}/{method}",
        {
            "page": 1,
            "size": total_size,
            "order": 0
        }
    )["data"]
    return pd.DataFrame(request_variant_info)


def list_magma_gene_by_trait_id(trait_id: str, genome: _Genome = "hg38") -> DataFrame:
    request_magma_data = request_get_data(f"detail/magma_gene/{trait_id}/{genome}")
    return pd.DataFrame(request_magma_data)


def list_magma_variant_info_data_by_trait_id(trait_id: str, genome: _Genome = "hg38") -> DataFrame:
    request_gene_variant_map_data = request_get_data(f"analysis/magma/gene/{trait_id}/{genome}")
    return pd.DataFrame(request_gene_variant_map_data)


def list_homer_tf_by_trait_id(trait_id: str, genome: _Genome = "hg38") -> DataFrame:
    request_homer_data = request_get_data(f"detail/homer_tf/{trait_id}/{genome}")
    return pd.DataFrame(request_homer_data)


def list_trait_gene_enrichment_data(
    trait_id: str,
    gene_set: _Gene_set = "GWAS_Catalog_2023",
    p_value: float = "1e-2",
    genome: _Genome = "hg38"
) -> DataFrame:
    request_gene_enrichment = request_post_data(
        "analysis/gene/enrichment",
        {
            "traitId": trait_id,
            "geneSet": gene_set,
            "value": p_value,
            "genome": genome
        }
    )
    return pd.DataFrame(request_gene_enrichment["traitGeneEnrichmentList"])
