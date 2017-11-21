from sklearn.cluster import KMeans
import numpy as np
from scipy import stats as st
from mnist import *
import pickle

train_images = read_idx('train-images.idx3-ubyte')
train_labels = read_idx('train-labels.idx1-ubyte')

index_lim = len(train_images) - 1
X = np.array([np.concatenate((train_images[x])) for x in range(0,index_lim)])

clustquant = 1000
kmeans = KMeans(n_clusters=clustquant, random_state=0).fit(X)

exec('pickle.dump( kmeans, open( "train_' + str(clustquant) + 'clust_' + str(index_lim) + 'imgs.p", "wb" ) )')