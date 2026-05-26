suppressPackageStartupMessages({
  if (!requireNamespace("Matrix", quietly = TRUE)) {
    stop("Missing R package 'Matrix'. Install it before running this script.")
  }
  if (!requireNamespace("monocle3", quietly = TRUE)) {
    stop("Missing R package 'monocle3'. Install it in R before running this script.")
  }
  if (!requireNamespace("SingleCellExperiment", quietly = TRUE)) {
    stop("Missing R package 'SingleCellExperiment'. Install it before running this script.")
  }
  if (!requireNamespace("SummarizedExperiment", quietly = TRUE)) {
    stop("Missing R package 'SummarizedExperiment'. Install it before running this script.")
  }
  if (!requireNamespace("igraph", quietly = TRUE)) {
    stop("Missing R package 'igraph'. Install it before running this script.")
  }
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) {
  stop(
    "Usage: Rscript run_paul15_monocle3.R ",
    "<mode: standard|injected> <input_dir> <output_csv> <root_cluster> ",
    "[embedding_csv] [graph_prefix] [cluster_k] [partition_qval] ",
    "[use_partition] [close_loop] [learn_graph_control]"
  )
}

mode <- args[[1]]
input_dir <- args[[2]]
output_csv <- args[[3]]
root_cluster <- args[[4]]
if (mode == "injected") {
  embedding_csv <- if (length(args) >= 5) args[[5]] else NA_character_
  graph_prefix <- if (length(args) >= 6) args[[6]] else NA_character_
  optional_start <- 7
} else {
  embedding_csv <- NA_character_
  graph_prefix <- if (length(args) >= 5) args[[5]] else NA_character_
  optional_start <- 6
}
if (!is.na(graph_prefix) && graph_prefix %in% c("", "NA", "None", "none")) {
  graph_prefix <- NA_character_
}

optional_arg <- function(position, default) {
  index <- optional_start + position - 1
  if (length(args) < index) {
    return(default)
  }
  value <- args[[index]]
  if (is.na(value) || value %in% c("", "NA", "None", "none")) {
    return(default)
  }
  value
}

parse_bool <- function(value) {
  if (is.logical(value)) {
    return(value)
  }
  lowered <- tolower(as.character(value))
  if (lowered %in% c("true", "t", "1", "yes", "y")) {
    return(TRUE)
  }
  if (lowered %in% c("false", "f", "0", "no", "n")) {
    return(FALSE)
  }
  stop("Cannot parse logical value: ", value)
}

parse_control_value <- function(value) {
  lowered <- tolower(value)
  if (lowered %in% c("true", "false", "t", "f")) {
    return(parse_bool(value))
  }
  numeric_value <- suppressWarnings(as.numeric(value))
  if (!is.na(numeric_value)) {
    return(numeric_value)
  }
  value
}

parse_learn_graph_control <- function(value) {
  if (is.na(value) || value %in% c("", "NA", "None", "none")) {
    return(NULL)
  }
  control <- list()
  pairs <- strsplit(value, ";", fixed = TRUE)[[1]]
  for (pair in pairs) {
    if (pair == "") {
      next
    }
    split_pair <- strsplit(pair, "=", fixed = TRUE)[[1]]
    if (length(split_pair) != 2) {
      stop("learn_graph_control entries must be key=value, got: ", pair)
    }
    control[[split_pair[[1]]]] <- parse_control_value(split_pair[[2]])
  }
  if (length(control) == 0) {
    return(NULL)
  }
  control
}

cluster_k <- as.integer(optional_arg(1, 20L))
partition_qval <- as.numeric(optional_arg(2, 0.05))
use_partition <- parse_bool(optional_arg(3, TRUE))
close_loop <- parse_bool(optional_arg(4, TRUE))
learn_graph_control <- parse_learn_graph_control(optional_arg(5, NA_character_))

if (!(mode %in% c("standard", "injected"))) {
  stop("mode must be 'standard' or 'injected'.")
}
if (mode == "injected" && (is.na(embedding_csv) || !file.exists(embedding_csv))) {
  stop("injected mode requires an existing embedding_csv.")
}

matrix_path <- file.path(input_dir, "expression_gene_by_cell.mtx")
cell_path <- file.path(input_dir, "cell_metadata.csv")
gene_path <- file.path(input_dir, "gene_metadata.csv")

expr <- Matrix::readMM(matrix_path)
cell_metadata <- read.csv(cell_path, stringsAsFactors = FALSE, check.names = FALSE)
gene_metadata <- read.csv(gene_path, stringsAsFactors = FALSE, check.names = FALSE)

if (!("cell_id" %in% colnames(cell_metadata))) {
  stop("cell_metadata.csv must contain a 'cell_id' column.")
}
if (!("gene_id" %in% colnames(gene_metadata))) {
  stop("gene_metadata.csv must contain a 'gene_id' column.")
}
if (!("gene_short_name" %in% colnames(gene_metadata))) {
  gene_metadata$gene_short_name <- gene_metadata$gene_id
}

rownames(cell_metadata) <- cell_metadata$cell_id
rownames(gene_metadata) <- gene_metadata$gene_id
rownames(expr) <- gene_metadata$gene_id
colnames(expr) <- cell_metadata$cell_id

cds <- monocle3::new_cell_data_set(
  expression_data = expr,
  cell_metadata = cell_metadata,
  gene_metadata = gene_metadata
)

set.seed(42)
cds <- monocle3::preprocess_cds(cds, num_dim = 50)

if (mode == "standard") {
  cds <- monocle3::reduce_dimension(
    cds,
    reduction_method = "UMAP",
    umap.n_neighbors = 15L,
    umap.min_dist = 0.1,
    umap.metric = "cosine"
  )
} else {
  embedding <- read.csv(embedding_csv, stringsAsFactors = FALSE, check.names = FALSE)
  if (!all(c("cell_id", "dim1", "dim2") %in% colnames(embedding))) {
    stop("embedding_csv must contain columns: cell_id, dim1, dim2.")
  }
  rownames(embedding) <- embedding$cell_id
  embedding <- embedding[colnames(cds), , drop = FALSE]
  if (any(is.na(embedding$dim1)) || any(is.na(embedding$dim2))) {
    stop("embedding_csv is missing coordinates for at least one cell.")
  }
  injected <- as.matrix(embedding[, c("dim1", "dim2")])
  colnames(injected) <- c("UMAP_1", "UMAP_2")
  reduced_dims <- SingleCellExperiment::reducedDims(cds)
  reduced_dims$UMAP <- injected
  SingleCellExperiment::reducedDims(cds) <- reduced_dims
}

cds <- monocle3::cluster_cells(
  cds,
  reduction_method = "UMAP",
  k = cluster_k,
  partition_qval = partition_qval,
  random_seed = 42
)
cds <- monocle3::learn_graph(
  cds,
  use_partition = use_partition,
  close_loop = close_loop,
  learn_graph_control = learn_graph_control
)

clusters <- as.character(SummarizedExperiment::colData(cds)$paul15_clusters)
root_cells <- colnames(cds)[clusters == root_cluster]
if (length(root_cells) == 0) {
  warning(paste0("Root cluster '", root_cluster, "' not found; using the first cell as root."))
  root_cells <- colnames(cds)[1]
}
graph_aux <- cds@principal_graph_aux[["UMAP"]]
node_coords <- graph_aux$dp_mst
closest_vertex <- graph_aux$pr_graph_cell_proj_closest_vertex
root_pr_nodes <- character(0)
if (!is.null(closest_vertex)) {
  root_vertex_ids <- as.character(closest_vertex[root_cells, 1])
  root_vertex_counts <- sort(table(root_vertex_ids), decreasing = TRUE)
  root_pr_nodes <- names(root_vertex_counts)[1]
  if (!(root_pr_nodes %in% colnames(node_coords))) {
    prefixed_root_pr_nodes <- paste0("Y_", root_pr_nodes)
    if (prefixed_root_pr_nodes %in% colnames(node_coords)) {
      root_pr_nodes <- prefixed_root_pr_nodes
    }
  }
  message(
    "Selected root principal node for cluster ", root_cluster, ": ",
    root_pr_nodes, " (", as.integer(root_vertex_counts[[1]]), " / ",
    length(root_cells), " root-cluster cells)"
  )
}
if (!(length(root_pr_nodes) == 1 && root_pr_nodes %in% colnames(node_coords))) {
  stop("Could not identify a valid root principal node for cluster ", root_cluster, ".")
}
cds <- monocle3::order_cells(cds, root_pr_nodes = root_pr_nodes)

umap_coords <- SingleCellExperiment::reducedDims(cds)$UMAP
pseudotime <- monocle3::pseudotime(cds)

out <- data.frame(
  cell_id = colnames(cds),
  paul15_clusters = clusters,
  dim1 = umap_coords[, 1],
  dim2 = umap_coords[, 2],
  pseudotime = as.numeric(pseudotime),
  stringsAsFactors = FALSE
)
write.csv(out, output_csv, row.names = FALSE)
message("Saved Monocle 3 result: ", output_csv)

if (!is.na(graph_prefix)) {
  principal_graph <- cds@principal_graph[["UMAP"]]
  graph_aux <- cds@principal_graph_aux[["UMAP"]]
  node_coords <- graph_aux$dp_mst
  root_vertices <- root_pr_nodes

  nodes <- data.frame(
    node_id = colnames(node_coords),
    dim1 = as.numeric(node_coords[1, ]),
    dim2 = as.numeric(node_coords[2, ]),
    is_root = colnames(node_coords) %in% root_vertices,
    stringsAsFactors = FALSE
  )
  edges <- igraph::as_data_frame(principal_graph, what = "edges")
  write.csv(nodes, paste0(graph_prefix, "_nodes.csv"), row.names = FALSE)
  write.csv(edges, paste0(graph_prefix, "_edges.csv"), row.names = FALSE)
  message("Saved Monocle 3 principal graph: ", graph_prefix, "_nodes.csv / _edges.csv")
}
