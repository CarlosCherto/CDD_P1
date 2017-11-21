from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from math import *

X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])
kmeans = KMeans(n_clusters=6, random_state=0).fit(X)
print(kmeans.labels_)



kmeans.predict([[0, 0], [4, 4]])

print(kmeans.cluster_centers_)