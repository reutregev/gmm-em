import argparse

import numpy as np
import pandas as pd

from src.constants import N_CLUSTERS, MAX_ITER
from src.em_algorithm import GMM
from src.utils import get_item_with_max_value


def set_parser():
    parser = argparse.ArgumentParser(description="Implementation of GMM for clustering purpose")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--find_num_clusters_range", type=int, nargs=2,
                       help="Two numbers that specify the range of search for the number of clusters. Must be > 1")
    group.add_argument("--n_clusters", type=int, help="Use the given number of clusters (> 1) as a parameter for EM algorithm")
    parser.add_argument("--data_path", type=str,
                        help="Path to a binary file in NumPy .npy format, containing a array of [n_samples, n_features]")
    parser.add_argument("--max_iter", type=int, default=MAX_ITER, help="Max iterations for EM algorithm")

    args = parser.parse_args()
    # Input validation
    if args.find_num_clusters_range:
        if args.find_num_clusters_range[0] >= args.find_num_clusters_range[1]:
            raise argparse.ArgumentTypeError("--find_num_clusters_range must receive the smaller number first")
        if args.find_num_clusters_range[0] <= 1 or args.find_num_clusters_range[1] <= 1:
            raise argparse.ArgumentTypeError("--find_num_clusters_range must receive numbers greater than 1")
    else:
        if args.n_clusters <= 1:
            raise argparse.ArgumentTypeError("--n_clusters must receive a number greater than 1")

    return args


def main():
    args = set_parser()

    data = np.load(args.data_path)

    if args.n_clusters:
        gmm = GMM(args.n_clusters, args.max_iter, data)
        silhouette_score = gmm.run(plot_clusters=True, plot_ll=True)
        print(f"Silhouette score for {args.n_clusters} clusters: {silhouette_score:.4f}")

    else:
        silhouette_scores = {}
        for n_clusters in range(args.find_num_clusters_range[0], args.find_num_clusters_range[1] + 1):
            gmm = GMM(n_clusters, args.max_iter, data)
            silhouette_scores[n_clusters] = gmm.run(plot_clusters=False, plot_ll=False)
        best_clusters_fit, best_silhouette_score = get_item_with_max_value(silhouette_scores)
        print(f"Found {best_clusters_fit} clusters; Silhouette score: {best_silhouette_score:.4f}")


if __name__ == '__main__':
    main()
