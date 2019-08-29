#============================================================================================
# Name        : asym_linkage.pyx
# Author      : Samuel Marchal, Sebastian Szyller
# Version     : 1.0
# Copyright   : Copyright (C) Secure Systems Group, Aalto University {https://ssg.aalto.fi/}
# License     : This code is released under Apache 2.0 license
#============================================================================================

import numpy as np
cimport numpy as np

cdef extern from "numpy/npy_math.h":
    cdef enum:
        NPY_INFINITYF


cdef inline np.npy_int64 condensed_index(np.npy_int64 n, np.npy_int64 i, np.npy_int64 j):
    """
    Calculate the condensed index of element (i, j) in an n x n condensed
    matrix.
    """
    if i < j:
        return n * i - int(i * (i + 1) / 2) + (j - i - 1)
    elif i > j:
        return n * j - int(j * (j + 1) / 2) + (i - j - 1)


cdef inline np.npy_int64 vector_index(np.npy_int64 m, np.npy_int64 i, np.npy_int64 j):
    """
    Calculate the condensed index of element (i, j) in an m x n linear
    vector matrix. m is the number of columns, i is the row index and j the column index
    """
    return m * i + j


cdef class LinkageUnionFind:
    """Structure for fast cluster labeling in unsorted dendrogram."""
    cdef int[:] parent
    cdef int[:] size
    cdef int next_label

    def __init__(self, int n):
        self.parent = np.arange(2 * n - 1, dtype=np.intc)
        self.next_label = n
        self.size = np.ones(2 * n - 1, dtype=np.intc)

    cdef int merge(self, int x, int y):
        self.parent[x] = self.next_label
        self.parent[y] = self.next_label
        cdef int size = self.size[x] + self.size[y]
        self.size[self.next_label] = size
        self.next_label += 1
        return size

    cdef find(self, int x):
        cdef int p = x

        while self.parent[x] != x:
            x = self.parent[x]

        while self.parent[p] != x:
            p, self.parent[p] = self.parent[p], x

        return x


cdef label(double[:, :] Z, int n):
    """Correctly label clusters in unsorted dendrogram."""
    cdef LinkageUnionFind uf = LinkageUnionFind(n)
    cdef int i, x, y, x_root, y_root
    for i in range(n - 1):
        x, y = int(Z[i, 0]), int(Z[i, 1])
        x_root, y_root = uf.find(x), uf.find(y)
        if x_root < y_root:
            Z[i, 0], Z[i, 1] = x_root, y_root
        else:
            Z[i, 0], Z[i, 1] = y_root, x_root
        Z[i, 3] = uf.merge(x_root, y_root)


def mst_single_linkage_asym(double[:] dists, int n, int m):
    """Perform hierarchy clustering using MST algorithm for single linkage.
    Parameters
    ----------
    dists : ndarray
        A condensed matrix stores the pairwise distances of the observations.
    n : int
        The number of observations.
    m : int
        The number of randomly selected samples basis for distance computation
    Returns
    -------
    Z : ndarray, shape (n - 1, 4)
        Computed linkage matrix.
    """

    Z_arr = np.empty((n - 1, 4))
    cdef double[:, :] Z = Z_arr

    # Which nodes were already merged.
    cdef int[:] merged = np.zeros(n, dtype=np.intc)

    cdef double[:] D = np.empty(n)
    D[:] = NPY_INFINITYF

    cdef int i, k, x, y
    cdef double dist, current_min

    x = 0
    for k in range(1, n):
        current_min = NPY_INFINITYF
        merged[x] = 1

        for i in range(m):
            if i >= k:
               continue

            dist = dists[vector_index(m, k, i)] # to change no access to condensed elements
            if D[k] > dist:
                D[k] = dist

            if D[k] < current_min:
                y = i
                current_min = D[k]

        Z[k-1, 0] = k
        Z[k-1, 1] = y
        Z[k-1, 2] = current_min
        x = y

    # Sort Z by cluster distances.
    order = np.argsort(Z_arr[:, 2], kind='mergesort')
    Z_arr = Z_arr[order]
    # Find correct cluster labels and compute cluster sizes inplace.
    label(Z_arr, n)

    return Z_arr