import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from jupyterthemes import jtplot

#import scipy
from scipy import ndimage

from persim import plot_diagrams
from ripser import ripser, lower_star_img
from sklearn.cluster import KMeans

r""" Utils for my notebooks """


def plotSublevelMask(img, threshold):
    plt.figure(figsize=(10, 6))
    plt.xticks([])
    plt.yticks([])
    plt.title('Sublevel set in white: threshold = %.2f' % threshold)

    return plt.imshow((img <= threshold).astype(int), cmap='gray')


def plotPersistenceDiagrams(dgm, **args):
    """plotPersistenceDiagrams.  Runs plot_diagrams and restyles back the way
    it should stay.

    Parameters
    ----------
    dgm : numpy.array
        diagram to plot
    args :
        other options
    """
    plot_diagrams(dgm, **args)
    jtplot.style(ticks=True, grid=True, gridlines='--')  # Ugh


def plotKMeansModel(X, model):
    """plotKMeansModel. Scatter plot colored by kmeans label.

    Parameters
    ----------
    X : numpy.array
        dataset the model was fit on
    model : sklearn.cluster.KMeans
        a kmeans model
    """
    plt.figure()
    return plt.scatter(X[:, 0], X[:, 1], c=model.labels_)


def getObjectComponentIndexes(X, model):
    """getObjectComponents.  Get the 'components' (i.e., H_0-representatives)
    corresponding to the 'objects' we are trying to count.

    Parameters
    ----------
    X : numpy.array
        dataset the model was fit on
    model : sklearn.cluster.KMeans
        a kmeans model
    """
    centroids = findClusterCenters(X, model)
    topYIndex = centroids.argmax(axis=0)[1]
    indexMask = model.labels_ == topYIndex
    indexes = np.arange(model.labels_.shape[0])

    return indexes[indexMask]


def getObjectComponents(X, model):
    """getObjectComponents.  Get the 'components' (i.e., H_0-representatives)
    corresponding to the 'objects' we are trying to count.

    Parameters
    ----------
    X : numpy.array
        dataset the model was fit on
    model : sklearn.cluster.KMeans
        a kmeans model
    """
    return X[getObjectComponentIndexes(X, model)]


def findClusterCenters(X, model):
    numClusters = np.unique(model.labels_).shape[0]

    out = np.zeros((numClusters, 2))

    for i in np.unique(model.labels_):
        out[i, :] = X[model.labels_ == i, :].mean(axis=0)

    return out
