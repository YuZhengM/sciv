2. SCIV API
===========================

.. contents::
   :local:
   :depth: 4

File (file)
---------------------------

文件读写相关接口，用于处理单细胞 ATAC 数据、H5AD、H5 等格式文件。

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

Model (model)
---------------------------

模型核心接口，提供细胞类型关联分析和因果变异识别功能。

.. currentmodule:: sciv.model
.. autosummary::
   :toctree: generated/

   core
   association_score
   knock

Plot (plot)
---------------------------

可视化接口，包含多种图表类型用于数据分析和展示。

Graph
^^^^^^^^^^^^^^^^^^^^^^^^^^^

网络图可视化功能。

.. currentmodule:: sciv.plot
.. autosummary::
   :toctree: generated/

   graph
   communities_graph
   network_two_types

Heatmap
^^^^^^^^^^^^^^^^^^^^^^^^^^^

热图可视化功能。

.. currentmodule:: sciv.plot
.. autosummary::
   :toctree: generated/

   heatmap
   heatmap_annotation

Scatter
^^^^^^^^^^^^^^^^^^^^^^^^^^^

散点图可视化功能。

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

小提琴图可视化功能。

.. currentmodule:: sciv.plot
.. autosummary::
   :toctree: generated/

   violin_base
   violin_trait

Box
^^^^^^^^^^^^^^^^^^^^^^^^^^^

箱线图可视化功能。

.. currentmodule:: sciv.plot
.. autosummary::
   :toctree: generated/

   box_base
   box_trait

KDE
^^^^^^^^^^^^^^^^^^^^^^^^^^^

核密度估计图可视化功能。

.. currentmodule:: sciv.plot
.. autosummary::
   :toctree: generated/

   kde

Line
^^^^^^^^^^^^^^^^^^^^^^^^^^^

折线图可视化功能。

.. currentmodule:: sciv.plot
.. autosummary::
   :toctree: generated/

   base_line

Bar
^^^^^^^^^^^^^^^^^^^^^^^^^^^

柱状图可视化功能。

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

条形码图可视化功能。

.. currentmodule:: sciv.plot
.. autosummary::
   :toctree: generated/

   barcode_base
   barcode_trait

Pie
^^^^^^^^^^^^^^^^^^^^^^^^^^^

饼图可视化功能。

.. currentmodule:: sciv.plot
.. autosummary::
   :toctree: generated/

   base_pie
   pie_label
   pie_trait

Bubble
^^^^^^^^^^^^^^^^^^^^^^^^^^^

气泡图可视化功能。

.. currentmodule:: sciv.plot
.. autosummary::
   :toctree: generated/

   bubble

Radar
^^^^^^^^^^^^^^^^^^^^^^^^^^^

雷达图可视化功能。

.. currentmodule:: sciv.plot
.. autosummary::
   :toctree: generated/

   radar
   base_radar
   radar_trait

Venn
^^^^^^^^^^^^^^^^^^^^^^^^^^^

韦恩图可视化功能。

.. currentmodule:: sciv.plot
.. autosummary::
   :toctree: generated/

   two_venn
   three_venn

Core
^^^^^^^^^^^^^^^^^^^^^^^^^^^

核心绘图功能。

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

数据预处理接口，用于单细胞数据清洗、差异分析和富集分析。

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

Tool (tool)
---------------------------

工具函数接口，包含算法、矩阵操作和随机游走等核心计算功能。

Algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^

算法相关功能。

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

随机游走相关功能。

.. currentmodule:: sciv.tool
.. autosummary::
   :toctree: generated/

   trs_scale_norm
   TraitDataParallel
   random_walk
   RandomWalk

Matrix
^^^^^^^^^^^^^^^^^^^^^^^^^^^

矩阵操作相关功能。

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

Util (util)
---------------------------

通用工具接口，包含常量定义、日志记录和辅助函数。

Constant
^^^^^^^^^^^^^^^^^^^^^^^^^^^

常量定义。

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

核心辅助函数。

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
