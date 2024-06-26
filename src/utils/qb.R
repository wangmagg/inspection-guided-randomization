if (!require("tidyverse")) install.packages("tidyverse")
if (!require("quickblock")) install.packages("quickblock")
if (!require("optparse")) install.packages("optparse")
if (!require("roxygen2")) install.packages("roxygen2")

suppressPackageStartupMessages(library(tibble))
suppressPackageStartupMessages(library(quickblock))
suppressPackageStartupMessages(library(optparse))
suppressPackageStartupMessages(library(roxygen2))

sample_mult_z <- function(
    data_path,
    z_path,
    blocks_path,
    n_arms,
    n_accept,
    min_block_factor,
    seed) {

    #' Samples treatment allocations using threshold blocking design
    #'
    #' @param data_path Path to trial data
    #' @param z_path Path to save treatment allocations to
    #' @param save_blocks_path Path to save block assignments to
    #' @param n_arms Number of treatment arms
    #' @param n_per_arm Number of units per treatment arm
    #' @param n_accept Number of treatment allocations to sample
    #' @param min_block_factor Min block size as multiple of number of arms
    #' @return NULL
    #'
    # Sample allocations for each data replication

    set.seed(seed)
    data <- read.csv(file.path(data_path))

    # Set minimum block size
    min_block_size <- min_block_factor * n_arms

    # Compute distances between units
    dist <- distances::distances(data, normalize = "studentize")

    # Sample a pool of treatment allocations
    qb_blocks <- quickblock(dist, size_constraint = min_block_size)

    z_pool <- do.call(
        cbind,
        replicate(n_accept,
            as.numeric(assign_treatment(qb_blocks, treatments = 1:n_arms)),
            simplify = FALSE
        )
    )

    # Apply zero-based indexing for assignments and transpose
    z_pool <- t(z_pool - 1)

    # Save treatment allocations to file
    # save_fname <- sprintf("%d_QB_z.csv", data_iter)
    # save_blocks_fname <- sprintf("%d_QB_blocks.csv", data_iter)
    # save_subdir <- file.path(data_dir, "QB")
    # z_pool_path <- file.path(save_dir, save_fname)
    # qb_blocks_path <- file.path(save_dir, save_blocks_fname)

    write.csv(z_pool, z_path, row.names = FALSE)
    write.csv(qb_blocks, blocks_path, row.names = FALSE)
}


# Define command line arguments
option_list <- list(
    make_option(
        c("--data-path"),
        type = "character"),
    make_option(
        c("--z-path"),
        type = "character"),
    make_option(
        c("--blocks-path"),
        type = "character"),
    make_option(
        c("--n-arms"),
        type = "integer",
        default = 4),
    make_option(
        c("--n-accept"),
        type = "integer",
        default = 500),
    make_option(
        c("--min-block-factor"),
        type = "numeric",
        default = 2),
    make_option(
        c("--seed"),
        type = "integer",
        default = 42)
)

# Parse command line arguments
opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

# Sample treatment allocations
sample_mult_z(
    opt[["data-path"]],
    opt[["z-path"]],
    opt[["blocks-path"]],
    opt[["n-arms"]],
    opt[["n-accept"]],
    opt[["min-block-factor"]],
    opt[["seed"]]
)