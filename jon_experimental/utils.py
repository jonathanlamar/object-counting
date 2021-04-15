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


def plotSuperlevelMask(img, threshold):
    plt.figure(figsize=(10, 6))
    plt.xticks([])
    plt.yticks([])
    plt.title('Sublevel set in white: threshold = %.2f' % threshold)

    return plt.imshow((img >= threshold).astype(int), cmap='gray')


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


def getLabeledDataset(X, model):
    return pd.DataFrame({
        'x': X[:, 0],
        'y': X[:, 1],
        'label': model.labels_
    })


def plotClusterModel(df):
    """plotClusterModel. Scatter plot colored by cluster label.

    Parameters
    ----------
    df : pandas.DataFrame
       output of getLabeledDataset
    """
    plt.figure()

    for c in df['label'].unique():
        df2 = df[df['label'] == c]
        labelStr = 'grp %d, y=%.2f, num=%d' % (
            c, df2['y'].mean(), df2['y'].shape[0])
        plt.scatter(df2['x'], df2['y'], label=labelStr)

    return plt.legend()


def getObjectComponentIndexes(df):
    """getObjectComponents.  Get the 'components' (i.e., H_0-representatives)
    corresponding to the 'objects' we are trying to count.

    Parameters
    ----------
    df : pandas.DataFrame
       output of getLabeledDataset
    """
    centroids = findClusterCenters(df)
    topYIndex = centroids.sort_values('y').index[-1]

    return df[df['label'] == topYIndex].index


def getObjectComponents(df):
    """getObjectComponents.  Get the 'components' (i.e., H_0-representatives)
    corresponding to the 'objects' we are trying to count.

    Parameters
    ----------
    df : pandas.DataFrame
       output of getLabeledDataset
    """
    return df.loc[getObjectComponentIndexes(df)]


def findClusterCenters(df):
    return df.groupby('label').mean()
