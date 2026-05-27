suppressPackageStartupMessages({library(Matrix); library(monocle3); library(SingleCellExperiment); library(SummarizedExperiment); library(igraph)})
input_dir <- "scripts/res/paul15/monocle3/monocle_input_no19lymph_no11dc"
embedding_csv <- "scripts/res/paul15/monocle3/raw/paul15_path_frozen_alpha0_embedding_k12_dg1_no19lymph_no11dc.csv"
expr <- Matrix::readMM(file.path(input_dir, "expression_gene_by_cell.mtx"))
cell_metadata <- read.csv(file.path(input_dir, "cell_metadata.csv"), stringsAsFactors = FALSE, check.names = FALSE)
gene_metadata <- read.csv(file.path(input_dir, "gene_metadata.csv"), stringsAsFactors = FALSE, check.names = FALSE)
rownames(cell_metadata) <- cell_metadata$cell_id; rownames(gene_metadata) <- gene_metadata$gene_id
rownames(expr) <- gene_metadata$gene_id; colnames(expr) <- cell_metadata$cell_id
cds <- monocle3::new_cell_data_set(expr, cell_metadata = cell_metadata, gene_metadata = gene_metadata)
set.seed(42); cds <- monocle3::preprocess_cds(cds, num_dim = 50)
embedding <- read.csv(embedding_csv, stringsAsFactors = FALSE, check.names = FALSE)
rownames(embedding) <- embedding$cell_id; embedding <- embedding[colnames(cds), , drop = FALSE]
injected <- as.matrix(embedding[, c("dim1", "dim2")]); colnames(injected) <- c("UMAP_1", "UMAP_2")
reducedDims(cds)$UMAP <- injected
cds <- monocle3::cluster_cells(cds, reduction_method="UMAP")
cds <- monocle3::learn_graph(cds, use_partition=TRUE)
aux <- cds@principal_graph_aux[["UMAP"]]
g <- cds@principal_graph[["UMAP"]]
cat("dp_mst dim names\n"); print(dim(aux$dp_mst)); print(rownames(aux$dp_mst)); print(head(colnames(aux$dp_mst)))
print(aux$dp_mst[,1:5])
cat("vertex names head\n"); print(head(V(g)$name)); print(length(V(g)$name)); print(all(V(g)$name %in% colnames(aux$dp_mst)))
cat("edge head\n"); print(head(as_data_frame(g, what="edges")))
