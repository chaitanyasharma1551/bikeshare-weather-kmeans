from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_blobs
from mpl_toolkits.mplot3d import Axes3D
# from scipy.spatial import distance
from scipy.spatial.distance import cdist, pdist


data = pd.read_csv('C:\\Users\\Chaitanya\\Downloads\\train.csv')
print(data.columns)
l = data.drop(columns = ['Unnamed: 0','date','events','Trips','total_docks','year','month','weekday'])
X = np.array(l)
y = np.array(data['Trips'])
for i in range(1,16):
    kmeans = KMeans(n_clusters=i)
    kmeans = kmeans.fit(X)
    print(kmeans.inertia_)

kmeans = KMeans(n_clusters=6)
kmeans = kmeans.fit(X)
C = kmeans.cluster_centers_

data['Assignments'] = kmeans.labels_
# print(data.columns)
newdf = data[['Assignments','Trips']]

out = newdf.groupby(['Assignments'], as_index= False).agg({'Trips':'mean'})
print(out)
mydf = pd.DataFrame(C,columns = l.columns)

mydf.to_csv("C:\\Users\\Chaitanya\\Desktop\\regression_input.csv")



