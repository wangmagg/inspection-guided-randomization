import numpy as np

def LinComb(scores_1, scores_2, w1, w2):
    if np.max(scores_1) == np.min(scores_1):
        scores_1_normed = scores_1
    else:
        scores_1_normed = (scores_1 - np.min(scores_1)) / (np.max(scores_1) - np.min(scores_1))
    if np.max(scores_2) == np.min(scores_2):
        scores_2_normed = scores_2
    else:
        scores_2_normed = (scores_2 - np.min(scores_2)) / (np.max(scores_2) - np.min(scores_2))

    return w1 * scores_1_normed + w2 * scores_2_normed

def get_agg(agg_name):
    if agg_name == "LinComb":
        agg_fn = LinComb
    else:
        raise ValueError(f"Unknown aggregation function: {agg_name}")
    return agg_fn
