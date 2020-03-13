from collections import defaultdict
import numpy as np
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array

from util import get_max_value_key, encode_features, get_unique_rows,decode_centroids, pandas_to_numpy
from util.dissim import matching_dissim, ng_dissim


def init_huang(X, n_clusters, dissim, random_state):
    n_attrs = X.shape[1]
    centroids = np.empty((n_clusters, n_attrs), dtype='object')
    for iattr in range(n_attrs):
        freq = defaultdict(int)
        for curattr in X[:, iattr]:
            freq[curattr] += 1
        choices = [chc for chc, wght in freq.items() for _ in range(wght)]
        # So that we are consistent between Python versions,
        # each with different dict ordering.
        choices = sorted(choices)
        centroids[:, iattr] = random_state.choice(choices, n_clusters)
    # The previously chosen centroids could result in empty clusters,
    # so set centroid to closest point in X.
    for ik in range(n_clusters):
        ndx = np.argsort(dissim(X, centroids[ik]))
        # We want the centroid to be unique, if possible.
        while np.all(X[ndx[0]] == centroids, axis=1).any() and ndx.shape[0] > 1:
            ndx = np.delete(ndx, 0)
        centroids[ik] = X[ndx[0]]

    return centroids


def init_cao(X, n_clusters, dissim):
    n_points, n_attrs = X.shape
    centroids = np.empty((n_clusters, n_attrs), dtype='object')
    # Method is based on determining density of points.
    dens = np.zeros(n_points)
    for iattr in range(n_attrs):
        freq = defaultdict(int)
        for val in X[:, iattr]:
            freq[val] += 1
        for ipoint in range(n_points):
            dens[ipoint] += freq[X[ipoint, iattr]] / float(n_points) / float(n_attrs)

    # Choose initial centroids based on distance and density.
    centroids[0] = X[np.argmax(dens)]
    if n_clusters > 1:
        # For the remaining centroids, choose maximum dens * dissim to the
        # (already assigned) centroid with the lowest dens * dissim.
        for ik in range(1, n_clusters):
            dd = np.empty((ik, n_points))
            for ikk in range(ik):
                dd[ikk] = dissim(X, centroids[ikk]) * dens
            centroids[ik] = X[np.argmax(np.min(dd, axis=0))]

    return centroids


def move_point_cat(point, ipoint, to_clust, from_clust, cl_attr_freq,
                   membship, centroids):
    membship[to_clust, ipoint] = 1
    membship[from_clust, ipoint] = 0
    # Update frequencies of attributes in cluster.
    for iattr, curattr in enumerate(point):
        to_attr_counts = cl_attr_freq[to_clust][iattr]
        from_attr_counts = cl_attr_freq[from_clust][iattr]

        # Increment the attribute count for the new "to" cluster
        to_attr_counts[curattr] += 1

        current_attribute_value_freq = to_attr_counts[curattr]
        current_centroid_value = centroids[to_clust][iattr]
        current_centroid_freq = to_attr_counts[current_centroid_value]
        if current_centroid_freq < current_attribute_value_freq:
            # We have incremented this value to the new mode. Update the centroid.
            centroids[to_clust][iattr] = curattr

        # Decrement the attribute count for the old "from" cluster
        from_attr_counts[curattr] -= 1

        old_centroid_value = centroids[from_clust][iattr]
        if old_centroid_value == curattr:
            # We have just removed a count from the old centroid value. We need to
            # recalculate the centroid as it may no longer be the maximum
            centroids[from_clust][iattr] = get_max_value_key(from_attr_counts)

    return cl_attr_freq, membship, centroids


def _labels_cost(X, centroids, dissim, membship=None):
    X = check_array(X)

    n_points = X.shape[0]
    cost = 0.
    labels = np.empty(n_points, dtype=np.uint16)
    for ipoint, curpoint in enumerate(X):
        diss = dissim(centroids, curpoint, X=X, membship=membship)
        clust = np.argmin(diss)
        labels[ipoint] = clust
        cost += diss[clust]

    return labels, cost


def _k_modes_iter(X, centroids, cl_attr_freq, membship, dissim, random_state):
    moves = 0
    for ipoint, curpoint in enumerate(X):
        clust = np.argmin(dissim(centroids, curpoint, X=X, membship=membship))
        if membship[clust, ipoint]:
            # Point is already in its right place.
            continue

        # Move point, and update old/new cluster frequencies and centroids.
        moves += 1
        old_clust = np.argwhere(membship[:, ipoint])[0][0]

        cl_attr_freq, membship, centroids = move_point_cat(
            curpoint, ipoint, clust, old_clust, cl_attr_freq, membship, centroids
        )

        # In case of an empty cluster, reinitialize with a random point
        # from the largest cluster.
        if not membship[old_clust, :].any():
            from_clust = membship.sum(axis=1).argmax()
            choices = [ii for ii, ch in enumerate(membship[from_clust, :]) if ch]
            rindx = random_state.choice(choices)

            cl_attr_freq, membship, centroids = move_point_cat(
                X[rindx], rindx, old_clust, from_clust, cl_attr_freq, membship, centroids
            )

    return centroids, moves


def k_modes_single(X, n_clusters, n_points, n_attrs, max_iter, dissim, init, init_no,
                   verbose, random_state):
    random_state = check_random_state(random_state)
    # _____ INIT _____
    if verbose:
        print("Init: initializing centroids")
    if isinstance(init, str) and init.lower() == 'huang':
        centroids = init_huang(X, n_clusters, dissim, random_state)
    elif isinstance(init, str) and init.lower() == 'cao':
        centroids = init_cao(X, n_clusters, dissim)
    elif isinstance(init, str) and init.lower() == 'random':
        seeds = random_state.choice(range(n_points), n_clusters)
        centroids = X[seeds]
    elif hasattr(init, '__array__'):
        # Make sure init is a 2D array.
        if len(init.shape) == 1:
            init = np.atleast_2d(init).T
        assert init.shape[0] == n_clusters, \
            "Wrong number of initial centroids in init ({}, should be {})." \
            .format(init.shape[0], n_clusters)
        assert init.shape[1] == n_attrs, \
            "Wrong number of attributes in init ({}, should be {})." \
            .format(init.shape[1], n_attrs)
        centroids = np.asarray(init, dtype=np.uint16)
    else:
        raise NotImplementedError

    if verbose:
        print("Init: initializing clusters")
    membship = np.zeros((n_clusters, n_points), dtype=np.uint8)
    # cl_attr_freq is a list of lists with dictionaries that contain the
    # frequencies of values per cluster and attribute.
    cl_attr_freq = [[defaultdict(int) for _ in range(n_attrs)]
                    for _ in range(n_clusters)]
    for ipoint, curpoint in enumerate(X):
        # Initial assignment to clusters
        clust = np.argmin(dissim(centroids, curpoint, X=X, membship=membship))
        membship[clust, ipoint] = 1
        # Count attribute values per cluster.
        for iattr, curattr in enumerate(curpoint):
            cl_attr_freq[clust][iattr][curattr] += 1
    # Perform an initial centroid update.
    for ik in range(n_clusters):
        for iattr in range(n_attrs):
            if sum(membship[ik]) == 0:
                # Empty centroid, choose randomly
                centroids[ik, iattr] = random_state.choice(X[:, iattr])
            else:
                centroids[ik, iattr] = get_max_value_key(cl_attr_freq[ik][iattr])

    # _____ ITERATION _____
    if verbose:
        print("Starting iterations...")
    itr = 0
    labels = None
    converged = False

    _, cost = _labels_cost(X, centroids, dissim, membship)

    epoch_costs = [cost]
    while itr <= max_iter and not converged:
        itr += 1
        centroids, moves = _k_modes_iter(
            X,
            centroids,
            cl_attr_freq,
            membship,
            dissim,
            random_state
        )
        # All points seen in this iteration
        labels, ncost = _labels_cost(X, centroids, dissim, membship)
        converged = (moves == 0) or (ncost >= cost)
        epoch_costs.append(ncost)
        cost = ncost
        if verbose:
            print("Run {}, iteration: {}/{}, moves: {}, cost: {}"
                  .format(init_no + 1, itr, max_iter, moves, cost))

    return centroids, labels, cost, itr, epoch_costs


def k_modes(X, n_clusters, max_iter, dissim, init, n_init, verbose, random_state, n_jobs):
    """k-modes algorithm"""
    random_state = check_random_state(random_state)
    if sparse.issparse(X):
        raise TypeError("k-modes does not support sparse data.")

    X = check_array(X, dtype=None)

    # Convert the categorical values in X to integers for speed.
    # Based on the unique values in X, we can make a mapping to achieve this.
    X, enc_map = encode_features(X)

    n_points, n_attrs = X.shape
    assert n_clusters <= n_points, "Cannot have more clusters ({}) " \
                                   "than data points ({}).".format(n_clusters, n_points)

    # Are there more n_clusters than unique rows? Then set the unique
    # rows as initial values and skip iteration.
    unique = get_unique_rows(X)
    n_unique = unique.shape[0]
    if n_unique <= n_clusters:
        max_iter = 0
        n_init = 1
        n_clusters = n_unique
        init = unique

    results = []
    seeds = random_state.randint(np.iinfo(np.int32).max, size=n_init)
    if n_jobs == 1:
        for init_no in range(n_init):
            results.append(k_modes_single(X, n_clusters, n_points, n_attrs, max_iter,
                                          dissim, init, init_no, verbose, seeds[init_no]))
    else:
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(k_modes_single)(X, n_clusters, n_points, n_attrs, max_iter,
                                    dissim, init, init_no, verbose, seed)
            for init_no, seed in enumerate(seeds))
    all_centroids, all_labels, all_costs, all_n_iters, all_epoch_costs = zip(*results)

    best = np.argmin(all_costs)
    if n_init > 1 and verbose:
        print("Best run was number {}".format(best + 1))

    return all_centroids[best], enc_map, all_labels[best], \
        all_costs[best], all_n_iters[best], all_epoch_costs[best]
