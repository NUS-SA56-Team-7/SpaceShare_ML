import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN,KMeans
from sklearn.decomposition import PCA
import data_generator.generate_searcher_data as generate_searcher_data
import pickle as Pickle
df = pd.read_csv("./dataset/cluster.csv",index_col=0)
scaler = PCA(2)
df = scaler.fit_transform(df.values)
cluster = KMeans(n_clusters=12)
label = cluster.fit_predict(df)
import matplotlib.pyplot as plt
u_labels = np.unique(label)  
for i in u_labels:
    plt.scatter(df[label == i , 0] , df[label == i , 1] , label = i)
plt.legend()
plt.show()
test = generate_searcher_data.generate_data(1,20)
print(test)
val = scaler.transform([test])
print(scaler.inverse_transform(cluster.cluster_centers_))
with open("./models/model.pkl",mode="wb") as f:
    Pickle.dump(cluster,f)
with open("./models/scaler.pkl",mode="wb") as f:
    Pickle.dump(scaler,f)