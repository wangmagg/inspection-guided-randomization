import numpy as np


def LinComb(scores_1: np.ndarray, scores_2: np.ndarray, w1: float, w2: float) -> float:
    """
    Apply max-min standardization, then linearly combine two sets of scores
    Args:
        - scores_1: Output from the first inspection metric
        - scores_2: Output from the second inspection metric
        - w1: Weight to apply to scores_1
        - w2: Weight to apply to scores_2
    """
    if np.max(scores_1) == np.min(scores_1):
        scores_1_normed = scores_1
    else:
        scores_1_normed = (scores_1 - np.min(scores_1)) / (
            np.max(scores_1) - np.min(scores_1)
        )
    if np.max(scores_2) == np.min(scores_2):
        scores_2_normed = scores_2
    else:
        scores_2_normed = (scores_2 - np.min(scores_2)) / (
            np.max(scores_2) - np.min(scores_2)
        )

    return w1 * scores_1_normed + w2 * scores_2_normed


def get_agg(agg_name: str):
    """
    Get aggregation function by name
    Args:
        - agg_name: Name of aggregation function
    """
    if agg_name == "LinComb":
        agg_fn = LinComb
    else:
        raise ValueError(f"Unknown aggregation function: {agg_name}")
    return agg_fn
