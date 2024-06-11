suppressPackageStartupMessages(library(tibble))
suppressPackageStartupMessages(library(quickblock))
suppressPackageStartupMessages(library(optparse))
suppressPackageStartupMessages(library(roxygen2))

load_data <- function(data_dir, data_rep, n_arms, n_per_arm) {
    #' Loads trial data
    #' 
    #' @param data_dir Directory containing trial data
    #' @param data_rep Data replication number
    #' @param n_arms Number of treatment arms
    #' @param n_per_arm Number of units per treatment arm
    #' @return Data frame containing trial data
    data_fname <- sprintf("%d.csv", data_rep)
    n_arms_subdir <- sprintf("arms-%d", n_arms)
    n_per_arm_subdir <- sprintf("n-per-arm-%d", n_per_arm)

    data_dir <- file.path(data_dir, n_arms_subdir, n_per_arm_subdir)
    read.csv(file.path(data_dir, data_fname))
}


sample_mult_z <- function(
    data_dir,
    n_data_reps,
    n_arms,
    n_per_arm,
    n_cutoff,
    min_block_factor,
    qb_dir) {

    #' Samples treatment allocations using threshold blocking design
    #' 
    #' @param data_dir Directory containing trial data
    #' @param n_data_reps Number of data replications
    #' @param n_arms Number of treatment arms
    #' @param n_per_arm Number of units per treatment arm
    #' @param n_cutoff Number of treatment allocations to sample
    #' @param min_block_factor Minimum block size as a multiple of the number of arms
    #' @param qb_dir Directory to save treatment allocations
    #' @return NULL
    
    # Sample allocations for each data replication
    for (data_rep in 1:n_data_reps) {
        data <- load_data(
            data_dir,
            data_rep - 1,
            n_arms,
            n_per_arm)

        # Set minimum block size
        min_block_size <- min_block_factor * n_arms

        # Compute distances between units
        dist <- distances::distances(data, normalize='studentize')

        # Sample a pool of treatment allocations
        qb_blocks <- quickblock(dist, size_constraint = min_block_size)

        z_pool <- do.call(
            cbind,
            replicate(n_cutoff,
                as.numeric(assign_treatment(qb_blocks, treatments = 1:n_arms)),
                simplify = FALSE
            )
        )

        # Apply zero-based indexing for assignments
        z_pool <- z_pool - 1

        # Save treatment allocations to file
        n_arms_subdir <- sprintf("arms-%d", n_arms)
        n_per_arm_subdir <- sprintf("n-per-arm-%d", n_per_arm)
        n_cutoff_subdir <- sprintf("n-cutoff-%d", n_cutoff)
        minblock_subdir <- sprintf("minblock-%d", min_block_factor)

        save_dir <- file.path(
            qb_dir, n_arms_subdir, n_per_arm_subdir,
            n_cutoff_subdir, minblock_subdir
        )
        
        if (!dir.exists(save_dir)) {
            dir.create(save_dir, recursive = TRUE)
        }
        save_fname <- sprintf("%d.csv", data_rep - 1)
        save_blocks_fname <- sprintf("%d-blocks.csv", data_rep - 1)

        write.csv(z_pool, file.path(save_dir, save_fname), row.names = FALSE)
        write.csv(qb_blocks, file.path(save_dir, save_blocks_fname), row.names = FALSE)
    }
}

setwd("/Users/maggiewang/Documents/stanford/repos/igr")

# Define command line arguments
option_list <- list(
    make_option(
        c("--data-dir"),
        type = "character",
        default = "data/mult-arm"),
    make_option(
        c("--n-arms"),
        type = "integer",
        default = 4),
    make_option(
        c("--n-per-arm"),
        type = "integer",
        default = 15),
    make_option(
        c("--n-cutoff"),
        type = "integer",
        default = 500),
    make_option(
        c("--min-block-factor"),
        type = "numeric",
        default = 2),
    make_option(
        c("--n-data-reps"),
        type = "integer",
        default = 4),
    make_option(
        c("--qb-dir"),
        type = "character",
        default = "data/mult-arm/qb")
)

# Parse command line arguments
opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

# Sample treatment allocations
sample_mult_z(
    opt[["data-dir"]],
    opt[["n-data-reps"]],
    opt[["n-arms"]],
    opt[["n-per-arm"]],
    opt[["n-cutoff"]],
    opt[["min-block-factor"]],
    opt[["qb-dir"]]
)