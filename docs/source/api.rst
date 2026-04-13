2. SCIV API
===========================

.. contents::
   :local:
   :depth: 1

File (.fl)
---------------------------

File read-write interface, used for processing single-cell ATAC data, H5AD, H5 and other format files.

.. currentmodule:: sciv.file
.. autosummary::
   :toctree: generated/

   barcodes_add_anno
   read_barcodes_file
   read_sc_atac
   read_sc_atac_10x_h5
   read_h5ad
   read_h5
   read_variants
   read_pkl
   to_meta
   to_fragments
   save_h5ad
   save_h5
   save_pkl

Model (.ml)
---------------------------

The core interface of the model provides functions for cell type association analysis and causal variation recognition.

.. currentmodule:: sciv.model
.. autosummary::
   :toctree: generated/

   core
   association_score
   knock

Plot (.pl)
---------------------------

Visual interface, including multiple chart types for data analysis and presentation.

Graph
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Network diagram visualization function.

.. currentmodule:: sciv.plot
.. autosummary::
   :toctree: generated/

   graph
   communities_graph
   network_two_types

Heatmap
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Heatmap visualization function.

.. currentmodule:: sciv.plot
.. autosummary::
   :toctree: generated/

   heatmap
   heatmap_annotation

Scatter
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Scatter chart visualization function.

.. currentmodule:: sciv.plot
.. autosummary::
   :toctree: generated/

   scatter_base
   scatter_3d
   scatter_atac
   scatter_trait
   volcano_base
   manhattan_causal_variant
   pseudo_time_score

Violin
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Violin chart visualization function.

.. currentmodule:: sciv.plot
.. autosummary::
   :toctree: generated/

   violin_base
   violin_trait

Box
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Visualization function of box diagram.

.. currentmodule:: sciv.plot
.. autosummary::
   :toctree: generated/

   box_base
   box_trait

KDE
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Visualization function of kernel density estimation map.

.. currentmodule:: sciv.plot
.. autosummary::
   :toctree: generated/

   kde

Line
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Line chart visualization function.

.. currentmodule:: sciv.plot
.. autosummary::
   :toctree: generated/

   base_line

Bar
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bar chart visualization function.

.. currentmodule:: sciv.plot
.. autosummary::
   :toctree: generated/

   bar
   bar_trait
   class_bar
   two_bar
   bar_significance

Barcode
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Barcode visualization function.

.. currentmodule:: sciv.plot
.. autosummary::
   :toctree: generated/

   barcode_base
   barcode_trait

Pie
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pie chart visualization function.

.. currentmodule:: sciv.plot
.. autosummary::
   :toctree: generated/

   base_pie
   pie_label
   pie_trait

Bubble
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bubble chart visualization function.

.. currentmodule:: sciv.plot
.. autosummary::
   :toctree: generated/

   bubble

Radar
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Radar visualization function.

.. currentmodule:: sciv.plot
.. autosummary::
   :toctree: generated/

   radar
   base_radar
   radar_trait

Venn
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Wayne diagram visualization function.

.. currentmodule:: sciv.plot
.. autosummary::
   :toctree: generated/

   two_venn
   three_venn

Core
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Core drawing function.

.. currentmodule:: sciv.plot
.. autosummary::
   :toctree: generated/

   trs_plot
   group_heatmap
   map_df_plot
   rate_bar_plot
   init_score_plot
   cell_cell_plot
   data_plot
   complete_ratio
   rate_circular_bar_plot

Preprocessing (preprocessing)
---------------------------

Data preprocessing interface, used for single-cell data cleaning, differential analysis, and enrichment analysis.

.. currentmodule:: sciv.preprocessing
.. autosummary::
   :toctree: generated/

   filter_data
   get_difference_genes
   get_difference_peaks
   paga_trajectory
   adata_map_df
   adata_group
   poisson_vi
   gsea_enrichr
   get_gene_enrichment
   get_sc_atac
   merge_sc_atac
   get_gene_expression
   get_peak_matrix
   get_tf_data

Tool (.tl)
---------------------------

Tool function interface, including core computing functions such as algorithms, matrix operations, and random walks.

Algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Algorithm related functions.

.. currentmodule:: sciv.tool
.. autosummary::
   :toctree: generated/

   sigmoid
   tf_idf
   z_score_normalize
   z_score_marginal
   marginal_normalize
   min_max_norm
   symmetric_scale
   mean_symmetric_scale
   coefficient_of_variation
   is_asc_sort
   lsi
   pca
   jaccard_similarity
   spectral_eigenmaps
   semi_mutual_knn_weight
   k_means
   spectral_clustering
   tsne
   umap
   kl_divergence
   safe_kl_divergence
   calinski_harabasz
   silhouette
   davies_bouldin
   ari
   ami
   binary_indicator
   z_score_to_p_value
   euclidean_distances
   overlap
   overlap_sum
   calculate_fragment_weighted_accessibility
   calculate_init_score_weight
   obtain_cell_cell_network
   perturb_data
   add_bernoulli_fluctuation_noise
   add_noise_perturb

Random Walk
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Random walk related functions.

.. currentmodule:: sciv.tool
.. autosummary::
   :toctree: generated/

   trs_scale_norm
   TraitDataParallel
   random_walk
   RandomWalk

Matrix
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Matrix operation related functions.

.. currentmodule:: sciv.tool
.. autosummary::
   :toctree: generated/

   split_matrix
   merge_matrix
   down_sampling_data
   matrix_dot_block_storage
   matrix_multiply_block_storage
   matrix_operation_memory_efficient
   vector_multiply_block_storage
   matrix_division_block_storage
   matrix_callback_block_storage

Util (.ul)
---------------------------

A universal tool interface that includes constant definitions, logging, and auxiliary functions.

Constant
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Constant definition.

.. currentmodule:: sciv.util
.. autosummary::
   :toctree: generated/

   project_version
   project_name
   project_cache_path
   is_form_log_file
   log_file_path
   path
   sparse_array
   sparse_matrix
   sparse_data
   dense_data
   matrix_data
   chrtype
   number
   collection
   enrichment_optional
   difference_peak_optional
   plot_color_types
   type_50_colors
   plot_cmap_50
   type_20_colors
   plot_cmap_20
   type_set_colors
   plot_cmap_set

Core
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Core auxiliary functions.

.. currentmodule:: sciv.util
.. autosummary::
   :toctree: generated/

   file_method
   log
   track_with_memory
   to_dense
   to_sparse
   sum_min_max
   get_index
   list_duplicate_set
   split_matrix
   merge_matrix
   list_index
   numerical_bisection_step
   get_real_predict_label
   strings_map_numbers
   set_inf_value
   plot_start
   plot_end
   generate_str
   check_adata_get
   add_cluster_info
   generate_hex_colors
   check_gpu_availability
