2. SCIV API
===========================

.. contents::
   :local:
   :depth: 1

File (.fl)
---------------------------

File read-write interface, used for processing single-cell ATAC data, H5AD, H5 and other format files.

.. automodule:: sciv.fl
   :members: barcodes_add_anno, read_barcodes_file, read_sc_atac, read_sc_atac_10x_h5, read_h5ad, read_h5, read_variants, read_pkl, to_meta, to_fragments, save_h5ad, save_h5, save_pkl
   :undoc-members:
   :show-inheritance:

Model (.ml)
---------------------------

The core interface of the model provides functions for cell type association analysis and causal variation recognition.

.. automodule:: sciv.ml
   :members: core, association_score, knock
   :undoc-members:
   :show-inheritance:

Plot (.pl)
---------------------------

Visual interface, including multiple chart types for data analysis and presentation.

Graph
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Network diagram visualization function.

.. automodule:: sciv.pl
   :members: graph, communities_graph, network_two_types
   :undoc-members:
   :show-inheritance:

Heatmap
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Heatmap visualization function.

.. automodule:: sciv.pl
   :members: heatmap, heatmap_annotation
   :undoc-members:
   :show-inheritance:

Scatter
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Scatter chart visualization function.

.. automodule:: sciv.pl
   :members: scatter_base, scatter_3d, scatter_atac, scatter_trait, volcano_base, manhattan_causal_variant, pseudo_time_score
   :undoc-members:
   :show-inheritance:

Violin
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Violin chart visualization function.

.. automodule:: sciv.pl
   :members: violin_base, violin_trait
   :undoc-members:
   :show-inheritance:

Box
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Visualization function of box diagram.

.. automodule:: sciv.pl
   :members: box_base, box_trait
   :undoc-members:
   :show-inheritance:

KDE
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Visualization function of kernel density estimation map.

.. automodule:: sciv.pl
   :members: kde
   :undoc-members:
   :show-inheritance:

Line
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Line chart visualization function.

.. automodule:: sciv.pl
   :members: base_line
   :undoc-members:
   :show-inheritance:

Bar
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bar chart visualization function.

.. automodule:: sciv.pl
   :members: bar, bar_trait, class_bar, two_bar, bar_significance, rate_bar_plot
   :undoc-members:
   :show-inheritance:

Barcode
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Barcode visualization function.

.. automodule:: sciv.pl
   :members: barcode_base, barcode_trait
   :undoc-members:
   :show-inheritance:

Pie
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pie chart visualization function.

.. automodule:: sciv.pl
   :members: base_pie, pie_label, pie_trait
   :undoc-members:
   :show-inheritance:

Bubble
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bubble chart visualization function.

.. automodule:: sciv.pl
   :members: bubble
   :undoc-members:
   :show-inheritance:

Radar
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Radar visualization function.

.. automodule:: sciv.pl
   :members: radar, base_radar, radar_trait, rate_circular_bar_plot
   :undoc-members:
   :show-inheritance:

Venn
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Wayne diagram visualization function.

.. automodule:: sciv.pl
   :members: two_venn, three_venn
   :undoc-members:
   :show-inheritance:

Preprocessing (.pp)
---------------------------

Data preprocessing interface, used for single-cell data cleaning, differential analysis, and enrichment analysis.

.. automodule:: sciv.pp
   :members: filter_data, get_difference_genes, get_difference_peaks, paga_trajectory, adata_map_df, adata_group, poisson_vi, gsea_enrichr, get_gene_enrichment, get_sc_atac, merge_sc_atac, get_gene_expression, get_peak_matrix, get_tf_data
   :undoc-members:
   :show-inheritance:

Tool (.tl)
---------------------------

Tool function interface, including core computing functions such as algorithms, matrix operations, and random walks.

Algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Algorithm related functions.

.. automodule:: sciv.tl
   :members: sigmoid, tf_idf, z_score_normalize, z_score_marginal, marginal_normalize, min_max_norm, symmetric_scale, mean_symmetric_scale, coefficient_of_variation, is_asc_sort, lsi, pca, jaccard_similarity, spectral_eigenmaps, semi_mutual_knn_weight, k_means, spectral_clustering, tsne, umap, kl_divergence, safe_kl_divergence, calinski_harabasz, silhouette, davies_bouldin, ari, ami, binary_indicator, z_score_to_p_value, euclidean_distances, overlap, overlap_sum, calculate_fragment_weighted_accessibility, calculate_init_score_weight, obtain_cell_cell_network, perturb_data, add_bernoulli_fluctuation_noise, add_noise_perturb
   :undoc-members:
   :show-inheritance:

Random Walk
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Random walk related functions.

.. automodule:: sciv.tl
   :members: trs_scale_norm, TraitDataParallel, random_walk, RandomWalk
   :undoc-members:
   :show-inheritance:

Matrix
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Matrix operation related functions.

.. automodule:: sciv.tl
   :members: split_matrix, merge_matrix, down_sampling_data, matrix_dot_block_storage, matrix_multiply_block_storage, matrix_operation_memory_efficient, vector_multiply_block_storage, matrix_division_block_storage, matrix_callback_block_storage
   :undoc-members:
   :show-inheritance:

Util (.ul)
---------------------------

A universal tool interface that includes constant definitions, logging, and auxiliary functions.

.. automodule:: sciv.ul
   :members: file_method, log, track_with_memory, to_dense, to_sparse, sum_min_max, get_index, list_duplicate_set, split_matrix, merge_matrix, list_index, numerical_bisection_step, get_real_predict_label, strings_map_numbers, set_inf_value, generate_str, check_adata_get, generate_hex_colors, check_gpu_availability
   :undoc-members:
   :show-inheritance:
