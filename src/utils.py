import operator
from typing import Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np


def plot_scatter_data(x1: np.ndarray, x2: np.ndarray, y: np.ndarray, iteration: int):
    plt.scatter(x1, x2, c=y, s=0.8)
    plt.title(f"Iteration {iteration}")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()


def plot_log_likelihood(ll_values: np.ndarray):
    plt.plot(ll_values, ".-")
    plt.title("Log-likelihood per Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("log-likelihood")
    plt.show()


def get_item_with_max_value(d: Dict[int, float]) -> Tuple[int, float]:
    """ Return an item with max value from a given dictionary """
    return max(d.items(), key=operator.itemgetter(1))
