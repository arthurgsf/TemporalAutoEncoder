# import statements
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
# from util import derivative
# from util import plot_curve
import sys
# import hierarchical clustering libraries
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import euclidean
# import dtw

# def dynamicTimeWarping(x, y):
#     d , _, _, _ = dtw.dtw(x, y, dist=euclidean)
#     return d


def n_clusters(data, plot=False, title="Dendogram"):
    Z = sch.linkage(data, method='single', metric=euclidean)
    f = Z[:, 2]
    max_dist = 0
    idx = 0
    for i in range(0, len(f) - 1):
        if f[i+1] - f[i]  > max_dist:
            max_dist = f[i+1]- f[i]
            idx = i+1
    n_clusters = data.shape[0] - idx
    if plot:
        sch.dendrogram(Z,p=10,truncate_mode='lastp', show_leaf_counts=True)
        plt.axhline(f[idx] - (f[idx] - f[idx - 1])/2)
        plt.title(title+ " dendogram")
        plt.text(0,f[idx] - (f[idx] - f[idx - 1])/2.2,"Optimal n_clusters = "+str(n_clusters))
        plt.savefig("../graficos/"+title+" dendogram.png", dpi='figure')
        plt.clf()
    return n_clusters

def get_labels(n_clusters, data):
    cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
    cluster.fit_predict(data)
    return cluster.labels_
    

# DATAPATH = "..\\bases_SISHCO\\clientes\\*.csv"
# with open("log.csv", "w") as f:
#     for s in glob.glob("../bases_SISHCO/*"):
#         s_name = s.split("/")[-1]
#         if(s_name == 'residencial'):
#             continue
#         print(s_name)
#         try:
#             dataset = pd.read_csv(s+ "/"+s_name+"_"+sys.argv[1]+".csv", sep=',').dropna(axis = 0)
#             f.write(s_name+","+str(n_clusters(dataset.values[:, 1:],plot=True,title=s_name+"_"+sys.argv[1]))+"\n")
#         except(FileNotFoundError):
#             print("Arquivo : " + s+ "/"+s_name+"_"+sys.argv[1]+".csv" + " não encontrado")
