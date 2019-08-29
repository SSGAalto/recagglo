#============================================================================================
# Name        : clustering.py
# Author      : Samuel Marchal, Sebastian Szyller
# Version     : 1.0
# Copyright   : Copyright (C) Secure Systems Group, Aalto University {https://ssg.aalto.fi/}
# License     : This code is released under Apache 2.0 license
#============================================================================================

from asym_linkage import mst_single_linkage_asym
import numpy as np
import pandas as pd
from scipy.spatial.distance import hamming, squareform
from scipy.cluster.hierarchy import linkage, fcluster


############# Utils functions #########################

def _convert_to_double(X):
    ''' Convert values of an input matrix to double
    '''
    if X.dtype != np.double:
        X = X.astype(np.double)
    if not X.flags.contiguous:
        X = X.copy()
    return X


def normalize_index(X, current_index):
    ''' Recompute cluster indexes for a new clustering based on the last cluster index (current_index)
    '''
    X[:, -1] = X[:, -1] + current_index
    new_index = np.max(X[:, -1])
    return X, new_index


def random_sample(X, n=0):
    ''' Randomly select n rows from an input matrix X
    Returns 2 matrices Y,Z:
        - one with the randomly sampled rows
        - the other being the input mat with randomly sampled rows removed
    '''

    nrows = X.shape[0]
    n = int(n)
    if n == 0:
        n = int(np.sqrt(nrows))
    idx = np.random.choice(X.shape[0], n, replace=False)
    return X[idx, :], np.delete(X, idx, 0)


############# Distance computation functions #########################

def dist(u, v, mode="hamming", weights=[]):
    ''' Compute the distance between two vectors
    of categorical values of same size
        Parameters
    ----------
    u : ndarray, shape (1, n)
    v : ndarray, shape (1, n)
    mode: string, chosen distance ("hamming" only option currently)
        Returns
    -------
    dist : float
        Computed distance.
    '''

    if mode == "hamming":
        if len(weights) == 0:
            return hamming(u, v)
        else:
            return hamming(u, v, weights)
    else:
        raise NotImplementedError("Distance function <{}> not implemented.".format(mode))


def distance_matrix(X, metric="hamming", weights=[]):
    ''' Compute the distance matrix for single input matrix
        Parameters
    ----------
    X : ndarray, shape (m, n)
        Returns
    -------
    Z : ndarray, shape (m, m)
        Computed distance matrix.
    '''

    size = X.shape[0]
    dist_X = np.zeros((size, size))
    for i in range(size):
        for j in range(i+1, size):
            dist_X[i, j] = dist(X[i], X[j], metric, weights)
    return dist_X + dist_X.T


def distance_matrix_2(X, Y, metric="hamming", weights=[]):
    ''' Compute the distance matrix for 2 input matrices
    First, compute distances between all rows of mat1 (reference matrix)
    Then, compute distances between rows of mat2 and rows of mat1
        Parameters
    ----------
    X : ndarray, shape (m, n)
    Y : ndarray, shape (o, n)
        Returns
    -------
    Z : ndarray, shape (m+o, m)
        Computed distance matrix.
    '''

    sizeX = X.shape[0]
    sizeY = Y.shape[0]
    dist_X = np.zeros((sizeX,sizeX))
    dist_Y = np.zeros((sizeY,sizeX))

    for i in range(sizeX):
        for j in range(i+1, sizeX):
            dist_X[i, j] = dist(X[i], X[j], metric, weights)
    dist_X = dist_X + dist_X.T

    for i in range(sizeY):
        for j in range(sizeX):
            dist_Y[i, j] = dist(Y[i], X[j], metric, weights)
    Z = np.concatenate((dist_X, dist_Y), axis=0)
    return Z


######## Clustering functions #################################

def AggloClust(c, linkage_mode='single', d_max=0.5, weights=[]):
    ''' Compute a new clustering for clusters in C using Recursive Agglomerative Clustering
    First, recursively split C in large clusters using SampleClust
    Second, split small clusters into clusters meeting maximum distance criteria d_max using AggloClust
        Parameters
    ----------
    c       : Pandas dataframe (contains elements to cluster)
    linkage_mode : the linkage method to use to generate cluster using AggloClust ('single','complete','average' or 'ward')
    d_max   : the maximum distance for cluster fusion
    ----------
        Returns
    Pandas dataframe of the computed clustering
    '''

    metric = "hamming"

    distance = distance_matrix(c, metric, weights)
    sparse_connectivity = squareform(distance)
    linkage_mat = linkage(sparse_connectivity, linkage_mode) #possible modes = 'single','complete','average','ward'
    cluster_res = fcluster(linkage_mat,d_max, 'distance')

    return np.append(c,cluster_res.reshape(cluster_res.shape[0], 1), axis=1)


def SampleClust(c, rho_s=1., rho_mc=6., weights=[]):
    ''' Compute a clustering for elements in c using Agglomerative Clustering with Sampling
        Parameters
    ----------
    c       : Pandas dataframe (contains elements to cluster)
    rho_s   : the multiplying factor for the sample size (multiplied by sqrt(|c|))
    rho_mc  : the dividing factor for maxclust number
    ----------
        Returns
    Pandas dataframe of the computed clustering
    '''

    metric = "hamming"
    n_samples = rho_s * np.sqrt(c.shape[0])
    Y,Z = random_sample(c,n_samples)

    distance = distance_matrix_2(Y, Z, metric, weights)
    m = distance.shape[0]
    n = distance.shape[1]
    linkage_mat = mst_single_linkage_asym(_convert_to_double(distance.flatten()),m,n)

    maxclust = m/rho_mc #the maximum number of clusters to generate

    cluster_res = fcluster(linkage_mat,maxclust,'maxclust')
    ordered_mat = np.append(Y,Z,axis=0)

    return np.append(ordered_mat,cluster_res.reshape(cluster_res.shape[0],1),axis=1) #may be merged those for results



def RecAgglo(C, delta_a=1000, delta_fc=1, d_max=0.5, rho_s=1, rho_mc=6, weights=[], verbose=False):
    ''' Compute a new clustering for clusters in C using Recursive Agglomerative Clustering
    First, recursively split C in large clusters using SampleClust
    Second, split small clusters into clusters meeting maximum distance criteria d_max using AggloClust
        Parameters
    ----------
    C       : Pandas dataframe (clustering to divide using RecAgglo)
    delta_a : the threshold value for using AggloClust
    delta_fc: the threshold value for stopping clustering
    d_max   : the maximum distance for cluster fusion
    rho_s   : the multiplying factor for the sample size
    rho_mc  : the dividing factor for maxclust number
    ----------
        Returns
    C_res   : Pandas dataframe (new clustering)
    '''

    treated = [] #list of row indices treated and already affected to clusters
    init = True

    # Loop to split existing clusters
    for index,count in pd.Series(C[:,-1]).value_counts().iteritems():
        if count > delta_a:                         # we use cluster sampling
            idx = np.where(C[:,-1] == index)
            tmp = C[idx]
            tmp_res = SampleClust(tmp[:,:-1],rho_s, rho_mc, weights)
            if len(pd.Series(tmp_res[:,-1]).value_counts()) > 1:        # we use recursive agglomerative clustering
                if verbose:
                    print("new Recursive Clustering", tmp.shape)

                treated = np.append(treated,idx)
                C_loop = RecAgglo(tmp_res,delta_a, delta_fc, d_max, rho_s,rho_mc, weights,verbose)
            elif rho_mc > 1.01:                     # we use alterantive recursive agglomerative clustering with low rho_mc
                if verbose:
                    print("Alterantive RecAgglo (rho_mc = 1.01)", tmp.shape)

                treated = np.append(treated,idx)
                C_loop = RecAgglo(tmp_res,delta_a, delta_fc, d_max, rho_s, 1.01, weights,verbose)
            elif count < delta_a * 4:               # fall back to agglomerative clustering
                if verbose:
                    print("Fall back AggloClust (4K)", tmp.shape)

                treated = np.append(treated,idx)
                C_loop = AggloClust(tmp[:,:-1],d_max=d_max,weights=weights)
            else:
                if verbose:                                   # No split possible / to re-cluster
                    print("Cannot cluster fall back (4K)", tmp.shape)

        elif count > delta_fc:                      # we use agglomerative clustering
            idx = np.where(C[:,-1] == index)
            tmp = C[idx]
            treated = np.append(treated,idx)
            C_loop = AggloClust(tmp[:,:-1],d_max=d_max, weights=weights)
        else:                                       # elements to re-cluster
            break

        if init and C_loop.shape[0] != 0:           # add new computed clusters to the final result
            C_res = C_loop
            last_index = np.max(C_res[:,-1]) + 1
            init = False
        elif C_loop.shape[0] != 0:
            to_add, last_index = normalize_index(C_loop,last_index)
            C_res = np.append(C_res,to_add,axis=0)


    if len(treated) > 0:                            # removing clustered elements from list to re-process
        remain = np.delete(C,treated.astype(int),0)
    else:
        remain = C
        if verbose:
            print("Nothing simplified")

    # clustering non-clustered elements in remain
    if remain.shape[0] > delta_a:                   # we use cluster sampling
        tmp_res = SampleClust(remain[:,:-1],rho_s, rho_mc, weights)
        if len(pd.Series(tmp_res[:,-1]).value_counts()) > 1:        # we use recursive agglomerative clustering
            C_end = RecAgglo(tmp_res,delta_a, delta_fc, d_max, rho_s, rho_mc,weights,verbose)
        else:                                       # elements remain as singletons
            if verbose:
                print("Sampling remaining failed", remain.shape)

            remain[:,-1] = np.arange(remain.shape[0])
            C_end = remain
    elif remain.shape[0] > delta_fc:                # we use agglomerative clustering
        C_end = AggloClust(remain[:,:-1],d_max=d_max,weights=weights)
    else:                                           # elements remain as singletins
        C_end = remain

    if init and C_end.shape[0] != 0:
        C_res = C_end
    elif C_end.shape[0] != 0:
        to_add, last_index = normalize_index(C_end,last_index)
        C_res = np.append(C_res,to_add,axis=0)

    if verbose:
        print("Loop completed with", C_res.shape)

    return C_res
