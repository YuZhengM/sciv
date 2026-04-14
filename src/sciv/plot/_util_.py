# -*- coding: UTF-8 -*-

import pandas as pd
from anndata import AnnData
from pandas import DataFrame
from tqdm import tqdm

from ..preprocessing import adata_map_df
from ..util import collection


def complete_ratio(
    adata: AnnData,
    layer: str = None,
    column: str = "value",
    extra_columns: collection = None,
    clusters: str = "clusters"
) -> DataFrame:
    """
    Calculate the completion ratio for enrichment analysis.

    This function computes the ratio of enriched vs non-enriched values for each
    trait-cluster combination. It handles missing data by filling in zero counts
    for combinations that don't exist in the original data.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object containing the data to be analyzed.
    layer : str, optional
        Specify the layer of the matrix to be processed. If None, uses the main matrix.
    column : str, default "value"
        The column name containing the binary enrichment values (1.0 for enriched, 0.0 for non-enriched).
    extra_columns : collection, optional
        Additional columns to include in the output DataFrame.
    clusters : str, default "clusters"
        The column name in adata.obs that defines the cell clusters.

    Returns
    -------
    DataFrame
        A DataFrame containing the completion ratio for each trait-cluster combination.
        Columns include: id, clusters, value, size_x, size_y, rate, and any extra_columns.
        The 'rate' column represents the ratio of enriched cells (value=1.0) to total cells.
    """
    # create data
    adata_df: DataFrame = adata_map_df(adata, column=column, layer=layer)

    clusters_group = adata_df.groupby(["id", clusters], as_index=False).size()
    value_group = adata_df.groupby(["id", clusters, column], as_index=False).size()
    new_value_group = value_group.merge(clusters_group, on=["id", clusters], how="left")

    if extra_columns is not None:
        extra_columns = list(extra_columns)
        extra_columns.extend(["id", clusters])
        new_value_group = new_value_group.merge(adata_df[extra_columns].drop_duplicates(), on=["id", clusters],
                                                how="left")

    # Completion
    id_list = list(set(new_value_group["id"]))
    clusters_list = list(set(new_value_group[clusters]))
    value_list = [1.0, 0.0]
    total_size = len(id_list) * len(clusters_list) * len(value_list)

    if total_size != new_value_group.shape[0]:
        new_value_group_index = (
            new_value_group["id"].astype(str) + "_"
            + new_value_group[clusters].astype(str) + "_"
            + new_value_group[column].astype(int).astype(str)
        )
        new_value_group.index = new_value_group_index
        new_value_group_index = list(new_value_group_index)

        trait_df: DataFrame = pd.DataFrame(columns=new_value_group.columns)

        # [id clusters  `column`  size_x size_y `extra_columns`]
        for _id_ in tqdm(id_list):
            for _clusters_ in clusters_list:
                for _value_ in value_list:

                    # At this point, it means that the enrichment effect is 1, while the non enrichment effect is 0,
                    # so it does not exist during grouping and needs to be added here
                    if (_id_ + "_" + _clusters_ + "_" + str(int(_value_))) not in new_value_group_index:
                        exit_value = 0 if int(_value_) == 1 else 1
                        exit_index = _id_ + "_" + _clusters_ + "_" + str(exit_value)
                        exit_data = new_value_group[new_value_group.index == exit_index]
                        exit_data.loc[exit_index, column] = _value_
                        exit_data.loc[exit_index, "size_x"] = 0
                        exit_data.index = [_id_ + "_" + _clusters_ + "_" + str(int(_value_))]
                        trait_df = pd.concat((trait_df, exit_data), axis=0)

        new_value_group = pd.concat((trait_df, new_value_group), axis=0)

    new_value_group["rate"] = new_value_group["size_x"] / new_value_group["size_y"]

    return new_value_group

