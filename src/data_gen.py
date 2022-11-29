from typing import Tuple

import numpy as np
from sklearn.datasets import make_blobs

from src.constants import CLUSTER_STD

np.random.seed(2)


def generate_data(n_samples: int, n_clusters: int, std: float = CLUSTER_STD) -> Tuple[np.ndarray, np.ndarray]:
    X, y = make_blobs(n_samples=n_samples,
                      centers=n_clusters,
                      cluster_std=std,
                      random_state=2)

    return X, y


def save_array(a: np.ndarray, path: str):
    np.save(path, a)
