import numpy as np


def _get_mirrors(z, n_trt_arms):
    mirrors = np.zeros((n_trt_arms, len(z)), dtype=np.float32)
    for i in range(n_trt_arms):
        mirrors[i] = z.copy()
        mirrors[i][z == (i + 1)] = 0
        mirrors[i][z == 0] = i + 1
    return mirrors

def _get_gfr_mirrors(z, same_rho_pairs):
    n_trt_arms = same_rho_pairs.shape[1] - 1
    mirrors = np.zeros((n_trt_arms, len(z)), dtype=np.float32)
    for i in range(n_trt_arms):
        mirrors[i] = z.copy()
        for pair in same_rho_pairs:
            mirrors[i][z == pair[1]] = pair[0]
            mirrors[i][z == pair[0]] = pair[1]
    return mirrors
    
def get_mirrors(z_pool_accepted, gfr=False, same_rho_pairs=None):
    if gfr:
        return np.vstack([_get_gfr_mirrors(z, same_rho_pairs) for z in z_pool_accepted])
    else:
        n_trt_arms = int(z_pool_accepted.max())
        return np.vstack([_get_mirrors(z, n_trt_arms) for z in z_pool_accepted])
        
def add_all_mirrors(
    z_pool_accepted, 
    z_pool_accepted_mirrors, 
    scores_1=None, 
    scores_mirrors_1=None, 
    scores_2=None,
    scores_mirrors_2=None
):  
    n_mirr_per_orig = z_pool_accepted_mirrors.shape[0] // z_pool_accepted.shape[0]   
    cutoff_idx_for_orig = z_pool_accepted.shape[0] // (n_mirr_per_orig + 1)
    cutoff_idx_for_mirr = cutoff_idx_for_orig * n_mirr_per_orig
    z_pool_accepted = np.vstack(
        [
            z_pool_accepted[:cutoff_idx_for_orig],
            z_pool_accepted_mirrors[:cutoff_idx_for_mirr],
        ]
    )
    print(f"Original: {cutoff_idx_for_orig}, Mirrors: {cutoff_idx_for_mirr}")
    if scores_1 is not None:
        scores_1 = np.hstack([scores_1[:cutoff_idx_for_orig], scores_mirrors_1[:cutoff_idx_for_mirr]])
    if scores_2 is not None:
        scores_2 = np.hstack([scores_2[:cutoff_idx_for_orig], scores_mirrors_2[:cutoff_idx_for_mirr]])
    return z_pool_accepted, (scores_1, scores_2)

def add_good_mirrors(
    z_pool_accepted,
    z_pool_accepted_mirrors,
    scores_1,
    scores_mirrors_1,
    scores_2=None,
    scores_mirrors_2=None,
    agg_fn=None,
    agg_kwargs=None
):
    n_accepted = z_pool_accepted.shape[0]
    map_mirr_to_orig_idx = np.repeat(np.arange(n_accepted), int(z_pool_accepted.max()))
    scores_all_1 = np.hstack([scores_1, scores_mirrors_1])
    if scores_2 is not None:
        scores_all_2 = np.hstack([scores_2, scores_mirrors_2])
        scores_all = agg_fn(scores_all_1, scores_all_2, **agg_kwargs)
    else:
        scores_all = scores_all_1

    max_accepted_score = np.max(scores_all[:n_accepted])
    mirrors_good_mask = scores_all[n_accepted:] <= max_accepted_score

    if sum(mirrors_good_mask) > 0:
        # Determine how many original and mirror allocations to keep based on n_cutoff
        z_pool_accepted_mirrors_good = z_pool_accepted_mirrors[mirrors_good_mask]
        map_mirr_to_orig_idx_good = map_mirr_to_orig_idx[mirrors_good_mask]
        bin_map = np.bincount(map_mirr_to_orig_idx_good)
        bin_map_csum = np.cumsum(bin_map)
        bin_map_add_orig_csum = np.cumsum(bin_map + 1)
        where_cutoff = np.where(bin_map_add_orig_csum >= n_accepted)

        if len(where_cutoff[0]) == 0:
            # all kept mirrors can be added
            cutoff_idx_for_mirr = bin_map_csum[-1]
            cutoff_idx_for_orig = n_accepted - cutoff_idx_for_mirr
        else:
            # only add mirrors up to the cutoff
            cutoff_idx_for_orig = where_cutoff[0][0]
            cutoff_idx_for_mirr = bin_map_csum[cutoff_idx_for_orig]

        z_pool_accepted = np.vstack(
            [
                z_pool_accepted[:cutoff_idx_for_orig],
                z_pool_accepted_mirrors_good[:cutoff_idx_for_mirr],
            ]
        )
        scores_1 = np.hstack([scores_1[:cutoff_idx_for_orig], scores_mirrors_1[:cutoff_idx_for_mirr]])
        if scores_2 is not None:
            scores_2 = np.hstack([scores_2[:cutoff_idx_for_orig], scores_mirrors_2[:cutoff_idx_for_mirr]])
        print(f"Original: {cutoff_idx_for_orig}, Mirrors: {cutoff_idx_for_mirr}")
    else:
        print("No accepted mirrors")

    return z_pool_accepted, (scores_1, scores_2)


def add_mirrors(
    mirror_type,
    z_pool_accepted,
    metric_1=None,
    scores_1=None,
    metric_2=None,
    scores_2=None,
    agg_fn=None,
    metric_1_kwargs=None,
    metric_2_kwargs=None,
    agg_kwargs=None,
    mirror_kwargs=None
):
    if mirror_kwargs is None:
        z_pool_accepted_mirrors = get_mirrors(z_pool_accepted)
    else:
        z_pool_accepted_mirrors = get_mirrors(z_pool_accepted, **mirror_kwargs)

    scores_mirrors_1 = None
    scores_mirrors_2 = None
    
    if metric_1 is not None:
        scores_mirrors_1 = metric_1(z_pool=z_pool_accepted_mirrors, **metric_1_kwargs)
    if metric_2 is not None:
        scores_mirrors_2 = metric_2(z_pool=z_pool_accepted_mirrors, **metric_2_kwargs)

    if mirror_type == "all":
        z_pool_accepted, (scores_1, scores_2) = add_all_mirrors(
            z_pool_accepted,
            z_pool_accepted_mirrors,
            scores_1,
            scores_mirrors_1,
            scores_2,
            scores_mirrors_2
        )
    elif mirror_type == "good":
        z_pool_accepted, (scores_1, scores_2) = add_good_mirrors(
            z_pool_accepted,
            z_pool_accepted_mirrors,
            scores_1,
            scores_mirrors_1,
            scores_2,
            scores_mirrors_2,
            agg_fn=agg_fn,
            agg_kwargs=agg_kwargs
        )
    else:
        raise ValueError(f"Invalid mirror type: {mirror_type}")
    
    return z_pool_accepted, (scores_1, scores_2)
