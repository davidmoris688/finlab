# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt
"""
Contains methods for verifying synthetic data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from mlfinlab.codependence import get_dependence_matrix
from mlfinlab.clustering.hierarchical_clustering import optimal_hierarchical_cluster


def plot_time_series_dependencies(time_series, dependence_method="gnpr_distance", **kwargs):
    """
    Plots the dependence matrix of a time series returns.

    Used to verify a time series' underlying distributions via the GNPR distance method.
    ``**kwargs`` are used to pass arguments to the `get_dependence_matrix` function used here.

    :param time_series: (pd.DataFrame) Dataframe containing time series.
    :param dependence_method: (str) Distance method to use by `get_dependence_matrix`
    :return: (plt.Axes) Figure's axes.
    """
    dep_matrix = get_dependence_matrix(time_series.diff().dropna(), dependence_method=dependence_method, **kwargs)

    # Plot dependence matrix
    fig, axis = plt.subplots(1, 1, figsize=(6, 5))
    plot = axis.pcolormesh(dep_matrix, cmap="viridis")
    fig.colorbar(plot, ax=axis)
    axis.set_title("Dependence Matrix using {} Metric".format(dependence_method))

    return axis

def _compute_eigenvalues(mats):
    """
    Computes the eigenvalues of each matrix.

    It is reproduced with modifications from the following blog post:
    `Marti, G. (2020) TF 2.0 DCGAN for 100x100 financial correlation matrices [Online].
    Available at: https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html.
    (Accessed: 17 Aug 2020)
    <https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html>`_

    :param mats: (np.array) List of matrices to calculate eigenvalues from.
        Has shape (n_sample, dim, dim)
    :return: (np.array) Resulting eigenvalues from mats.
    """
    eigenvalues = []
    for mat in mats:
        eigenvals, _ = np.linalg.eig(mat)
        eigenvalues.append(eigenvals)

    return np.array(eigenvalues)


def _compute_pf_vec(mats):
    """
    Computes the Perron-Frobenius vector of each matrix.

    It is reproduced with modifications from the following blog post:
    `Marti, G. (2020) TF 2.0 DCGAN for 100x100 financial correlation matrices [Online].
    Available at: https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html.
    (Accessed: 17 Aug 2020)
    <https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html>`_

    The Perron-Frobenius property asserts that for a strictly positive square matrix, the
    corresponding eigenvector of the largest eigenvalue has strictly positive components.

    :param mats: (np.array) List of matrices to calculate Perron-Frobenius vector from.
        Has shape (n_sample, dim, dim)
    :return: (np.array) Resulting Perron-Frobenius vectors from mats.
    """
    pf_vectors = []
    for mat in mats:
        # Calculate eigenvalues and eigenvectors. Append to the result the eigenvector
        # corresponding to the largest eigenvalue.
        eigenvals, eigenvecs = np.linalg.eig(mat)
        pf_vector = eigenvecs[:, np.argmax(eigenvals)]
        if pf_vector[0] < 0:
            pf_vector = -pf_vector
        pf_vectors.append(pf_vector)

    return np.array(pf_vectors)


def _compute_degree_counts(mats):
    """
    Computes the number of degrees in MST of each matrix.

    It is reproduced with modifications from the following blog post:
    `Marti, G. (2020) TF 2.0 DCGAN for 100x100 financial correlation matrices [Online].
    Available at: https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html.
    (Accessed: 17 Aug 2020)
    <https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html>`_

    The degree count is calculated by computing the MST of the matrix, and counting
    how many times each nodes appears in each edge produced by the MST. This count is normalized
    by the size of the matrix.

    :param mats: (np.array) List of matrices to calculate the number of degrees in MST from.
        Has shape (n_sample, dim, dim)
    :return: (np.array) Resulting number of degrees in MST from mats.
    """
    all_counts = []
    for mat in mats:
        # Compute MST.
        dist = (1 - mat) / 2
        graph = csr_matrix(dist)
        mst = minimum_spanning_tree(graph)

        # Count number of degrees.
        degrees = {i: 0 for i in range(len(mat))}
        for edge in np.argwhere(mst.todense() != 0):
            degrees[edge[0]] += 1
            degrees[edge[1]] += 1
        degrees = pd.Series(degrees).sort_values(ascending=False)
        cur_counts = degrees.value_counts()

        # Save the counts.
        counts = np.zeros(len(mat))
        for i in range(len(mat)):
            if i in cur_counts:
                counts[i] = cur_counts[i]

        all_counts.append(counts / (len(mat) - 1))

    return np.array(all_counts)


def plot_pairwise_dist(emp_mats, gen_mats, n_hist=100):
    """
    Plots the following stylized facts for comparison between empirical and generated
    correlation matrices:

    - Distribution of pairwise correlations is significantly shifted to the positive.

    It is reproduced with modifications from the following blog post:
    `Marti, G. (2020) TF 2.0 DCGAN for 100x100 financial correlation matrices [Online].
    Available at: https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html.
    (Accessed: 17 Aug 2020)
    <https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html>`_

    :param emp_mats: (np.array) Empirical correlation matrices.
        Has shape (n_samples_a, dim_a, dim_a)
    :param gen_mats: (np.array) Generated correlation matrices.
        Has shape (n_samples_b, dim_b, dim_b)
    :param n_hist: (int) Number of bins for histogram plots. (100 by default).
    :return: (plt.Axes) Figure's axes.
    """
    tri_rows, tri_cols = np.triu_indices(emp_mats.shape[1], k=1)
    _, axes = plt.subplots(1, 1, figsize=(8, 4))
    axes.hist(
        gen_mats[:, tri_rows, tri_cols].flatten(),
        bins=n_hist,
        alpha=0.5,
        color="b",
        density=True,
        label="Synthetic",
    )
    axes.hist(
        emp_mats[:, tri_rows, tri_cols].flatten(),
        bins=n_hist,
        alpha=0.5,
        color="r",
        density=True,
        label="Empirical",
    )
    axes.axvline(x=np.mean(gen_mats), color="b", linestyle="dashed", linewidth=2)
    axes.axvline(x=np.mean(emp_mats), color="r", linestyle="dashed", linewidth=2)
    axes.legend()
    axes.set_title(
        "Pairwise Correlations Distribution\nIt must be significantly shifted to the positive"
    )

    return axes


def plot_eigenvalues(emp_mats, gen_mats, n_hist=100):
    """
    Plots the following stylized facts for comparison between empirical and generated
    correlation matrices:

    - Eigenvalues follow the Marchenko-Pastur distribution, but for a very large first eigenvalue (the market).

    - Eigenvalues follow the Marchenko-Pastur distribution, but for a couple of other large eigenvalues (industries).

    It is reproduced with modifications from the following blog post:
    `Marti, G. (2020) TF 2.0 DCGAN for 100x100 financial correlation matrices [Online].
    Available at: https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html.
    (Accessed: 17 Aug 2020)
    <https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html>`_

    :param emp_mats: (np.array) Empirical correlation matrices.
        Has shape (n_samples_a, dim_a, dim_a)
    :param gen_mats: (np.array) Generated correlation matrices.
        Has shape (n_samples_b, dim_b, dim_b)
    :param n_hist: (int) Number of bins for histogram plots. (100 by default).
    :return: (plt.Axes) Figure's axes.
    """
    # Get mean eigenvalues.
    mean_emp_eigenvals = np.mean(_compute_eigenvalues(emp_mats), axis=0)
    mean_gen_eigenvals = np.mean(_compute_eigenvalues(gen_mats), axis=0)

    _, axes = plt.subplots(1, 1, figsize=(8, 4))
    axes.hist(
        mean_gen_eigenvals, bins=n_hist, color="b", density=True, alpha=0.5, label="Synthetic"
    )
    axes.hist(
        mean_emp_eigenvals, bins=n_hist, color="r", density=True, alpha=0.5, label="Emprical"
    )
    axes.legend()
    axes.set_title(
        "Mean Eigenvalues\nMust have very large first eigenvalue,\nfollowed for a couple of "
        "other large eigenvalues"
    )

    return axes


def plot_eigenvectors(emp_mats, gen_mats, n_hist=100):
    """
    Plots the following stylized facts for comparison between empirical and generated
    correlation matrices:

    - Perron-Frobenius property (first eigenvector has positive entries).

    It is reproduced with modifications from the following blog post:
    `Marti, G. (2020) TF 2.0 DCGAN for 100x100 financial correlation matrices [Online].
    Available at: https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html.
    (Accessed: 17 Aug 2020)
    <https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html>`_

    :param emp_mats: (np.array) Empirical correlation matrices.
       Has shape (n_samples_a, dim_a, dim_a)
    :param gen_mats: (np.array) Generated correlation matrices.
       Has shape (n_samples_b, dim_b, dim_b)
    :param n_hist: (int) Number of bins for histogram plots. (100 by default).
    :return: (plt.Axes) Figure's axes.
    """
    # Get mean eigenvectors.
    mean_emp_pf = np.mean(_compute_pf_vec(emp_mats), axis=0)
    mean_gen_pf = np.mean(_compute_pf_vec(gen_mats), axis=0)

    _, axes = plt.subplots(1, 1, figsize=(8, 4))
    axes.hist(mean_gen_pf, bins=n_hist, density=True, color="b", alpha=0.5, label="Synthetic")
    axes.hist(mean_emp_pf, bins=n_hist, density=True, color="r", alpha=0.5, label="Empirical")
    axes.axvline(x=0, color="k", linestyle="dashed", linewidth=2)
    axes.axvline(x=np.mean(mean_gen_pf), color="b", linestyle="dashed", linewidth=2)
    axes.axvline(x=np.mean(mean_emp_pf), color="r", linestyle="dashed", linewidth=2)
    axes.legend()
    axes.set_title(
        "Eigenvector Entries follow Perron-Frobenius Property\nFirst eigenvector has positive "
        "entries"
    )

    return axes


def plot_hierarchical_structure(emp_mats, gen_mats):
    """
    Plots the following stylized facts for comparison between empirical and generated
    correlation matrices:

    - Hierarchical structure of correlations.

    It is reproduced with modifications from the following blog post:
    `Marti, G. (2020) TF 2.0 DCGAN for 100x100 financial correlation matrices [Online].
    Available at: https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html.
    (Accessed: 17 Aug 2020)
    <https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html>`_

    :param emp_mats: (np.array) Empirical correlation matrices.
       Has shape (n_samples_a, dim_a, dim_a)
    :param gen_mats: (np.array) Generated correlation matrices.
       Has shape (n_samples_b, dim_b, dim_b)
    :return: (tuple) Figures' axes.
    """
    tri_rows, tri_cols = np.triu_indices(emp_mats.shape[1], k=1)

    # Arrange with hierarchical clustering by maximizing the sum of the
    # similarities between adjacent leaves.
    corr_mat = emp_mats[0]
    dist = 1 - corr_mat
    linkage_mat = hierarchy.linkage(dist[tri_rows, tri_cols], method="ward")
    optimal_leaves = hierarchy.optimal_leaf_ordering(linkage_mat, dist[tri_rows, tri_cols])
    optimal_ordering = hierarchy.leaves_list(optimal_leaves)
    ordered_corr = corr_mat[optimal_ordering, :][:, optimal_ordering]

    fig, (axes1, axes2) = plt.subplots(2, 1, figsize=(5, 7))
    plot_1 = axes1.pcolormesh(ordered_corr, cmap="viridis")
    fig.colorbar(plot_1, ax=axes1)
    axes1.set_title("Empirical Matrix Hierarchical Structure")
    plot_2 = axes2.pcolormesh(gen_mats[0], cmap="viridis")
    fig.colorbar(plot_2, ax=axes2)
    axes2.set_title("Synthetic Matrix Hierarchical Structure")

    return axes1, axes2


def plot_mst_degree_count(emp_mats, gen_mats):
    """
    Plots all the following stylized facts for comparison between empirical and generated
    correlation matrices:

    - Scale-free property of the corresponding Minimum Spanning Tree (MST).

    It is reproduced with modifications from the following blog post:
    `Marti, G. (2020) TF 2.0 DCGAN for 100x100 financial correlation matrices [Online].
    Available at: https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html.
    (Accessed: 17 Aug 2020)
    <https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html>`_

    :param emp_mats: (np.array) Empirical correlation matrices.
       Has shape (n_samples_a, dim_a, dim_a)
    :param gen_mats: (np.array) Generated correlation matrices.
       Has shape (n_samples_b, dim_b, dim_b)
    :return: (plt.Axes) Figure's axes.
    """
    # Get mean MST degree counts.
    mean_corrgan_counts = np.mean(_compute_degree_counts(gen_mats), axis=0)
    mean_corrgan_counts = pd.Series(mean_corrgan_counts).replace(0, np.nan)
    mean_empirical_counts = np.mean(_compute_degree_counts(emp_mats), axis=0)
    mean_empirical_counts = pd.Series(mean_empirical_counts).replace(0, np.nan)

    _, axes = plt.subplots(1, 1, figsize=(8, 4))
    axes.scatter(mean_corrgan_counts.index, mean_corrgan_counts, label="Synthetic")
    axes.scatter(mean_empirical_counts.index, mean_empirical_counts, label="Empirical")
    axes.legend()
    axes.set_title("MST Degree Count\nCount must exhibit the scale-free property")

    return axes


def plot_stylized_facts(emp_mats, gen_mats, n_hist=100):
    """
    Plots the following stylized facts for comparison between empirical and generated
    correlation matrices:

    1. Distribution of pairwise correlations is significantly shifted to the positive.

    2. Eigenvalues follow the Marchenko-Pastur distribution, but for a very large first
    eigenvalue (the market).

    3. Eigenvalues follow the Marchenko-Pastur distribution, but for a couple of other
    large eigenvalues (industries).

    4. Perron-Frobenius property (first eigenvector has positive entries).

    5. Hierarchical structure of correlations.

    6. Scale-free property of the corresponding Minimum Spanning Tree (MST).

    It is reproduced with modifications from the following blog post:
    `Marti, G. (2020) TF 2.0 DCGAN for 100x100 financial correlation matrices [Online].
    Available at: https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html.
    (Accessed: 17 Aug 2020)
    <https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html>`_

    :param emp_mats: (np.array) Empirical correlation matrices.
        Has shape (n_samples_a, dim_a, dim_a)
    :param gen_mats: (np.array) Generated correlation matrices.
        Has shape (n_samples_b, dim_b, dim_b)
    :param n_hist: (int) Number of bins for histogram plots. (100 by default).
    """
    # Plot distribution of pairwise correlations.
    plot_pairwise_dist(emp_mats, gen_mats, n_hist)

    # Plot eigenvalues to check they follow the Marchenko-Pastur distribution.
    plot_eigenvalues(emp_mats, gen_mats, n_hist)

    # Plot eigenvector entries to check Perron-Frobenius property.
    plot_eigenvectors(emp_mats, gen_mats, n_hist)

    # Plot hierarchical structure of correlations.
    plot_hierarchical_structure(emp_mats, gen_mats)

    # Plot MST degree counts.
    plot_mst_degree_count(emp_mats, gen_mats)

    plt.show()


def plot_optimal_hierarchical_cluster(mat, method="ward"):
    """
    Calculates and plots the optimal clustering of a correlation matrix.

    It uses the `optimal_hierarchical_cluster` function in the clustering module to calculate
    the optimal hierarchy cluster matrix.

    :param mat: (np.array/pd.DataFrame) Correlation matrix.
    :param method: (str) Method to calculate the hierarchy clusters. Can take the values
        ["single", "complete", "average", "weighted", "centroid", "median", "ward"].
    :return: (plt.Axes) Figure's axes.
    """
    ordered_corr = optimal_hierarchical_cluster(mat, method)
    fig, axes = plt.subplots(1, 1, figsize=(6, 4))
    plot_1 = axes.pcolormesh(ordered_corr, cmap="viridis")
    fig.colorbar(plot_1, ax=axes)

    return axes
