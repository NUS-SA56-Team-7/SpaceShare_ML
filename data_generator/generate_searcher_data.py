import types
import numpy as np
import pandas as pd
import random as rd
import uuid
from sklearn.preprocessing import OrdinalEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle
import json
import os
ROOM_TYPE = ["SINGLE","MASTER","COMMON","WHOLE_UNIT"]
PROPERTY_TYPE = ["HDB","CONDOMINIUM","LANDED"]
def seacher_data(n,id):
    data = [[id,uuid.uuid1(),PROPERTY_TYPE[rd.randint(0,2)],ROOM_TYPE[rd.randint(0,3)]] for i in range(n)]
    return pd.DataFrame(data=data,columns=["tenantId","propertyId","propertyType","roomType"])
def generate_data(n=None,id=None,df=None):
    if isinstance(df, types.NoneType) == True:
        df= seacher_data(n,id)
    property_type = [0]*3
    room_type = [0]*4
    for x in np.unique(df["propertyType"].values):
        property_type[x] = df["propertyType"].value_counts()[x]
    for x in np.unique(df["roomType"].values):
        room_type[x] = df["roomType"].value_counts()[x]
    return property_type+room_type
def df_to_freq(df,id,property_unique=[0,1,2],room_unique=[0,1,2,3]):
    freq_room = [0]*4
    freq_property =[0]*3
    count_property = df["property_type"].value_counts()
    for unique in property_unique:
        try:
            freq_property[int(unique)]=count_property[int(unique)]
        except:
            pass
    freq_property = [x/len(df) for x in freq_property]
    count_room = df["room_type"].value_counts()
    for unique in room_unique:
        try:
            freq_room[int(unique)]=count_room[int(unique)]
        except:
            pass
    freq_room = [x/len(df) for x in freq_room]
    return [id]+freq_property+freq_room
def data_to_cluster(df,label):
    freqs = []
    for i in np.unique(df['tenant_id']):
        freqs.append(df_to_freq(df=df[df["tenant_id"]==i],id=i))
    df_freq=pd.DataFrame(data=freqs,
                        columns=["tenantId","CONDOMINIUM","HDB","LANDED","COMMON","MASTER","SINGLE","WHOLE_UNIT"]
                        ).set_index("tenantId")
    datas = [list(x) for x in zip(df_freq.index.tolist(),label)]
    return pd.DataFrame(data=datas,columns=["id","cluster"])

def main():
    dfs = []
    for i in range(1000):
        df = seacher_data(rd.randint(1,20),i)
        dfs.append(df)
    df_final = pd.concat(dfs,ignore_index=True)
    df_final.to_csv("./dataset/mock_data.csv")
    df_property_list = df_final[["propertyId","propertyIype","roomType"]]
    df_property_list.to_csv("./dataset/mock_property.csv")
    encoder = OrdinalEncoder()
    df_property_list[["property_type","room_type"]] = encoder.fit_transform(df_property_list[["property_type","room_type"]])
    df_property_list=df_property_list.set_index("property_id")
    print(df_property_list)
    df_final[["property_type","room_type"]]=encoder.transform(df_final[["property_type","room_type"]])
    print(df_final)
    property_unique = np.unique(df_final["property_type"])
    room_unique= np.unique(df_final["room_type"])
    freqs = []
    for i in np.unique(df_final["tenant_id"]):   
        freqs.append(df_to_freq(df=df_final[df_final["tenant_id"]==i],id=i))
    df_freq=pd.DataFrame(data=freqs,
                        columns=["tenant_id","CONDOMINIUM","HDB","LANDED","COMMON","MASTER","SINGLE","WHOLE_UNIT"]
                        ).set_index("tenant_id")
    print(df_freq)
    pc = PCA(2)
    cluster = KMeans(n_clusters=12)
    df_fit = pc.fit_transform(df_freq)
    print(df_fit)
    label = cluster.fit_predict(df_fit)
    test = cluster.predict(pc.transform([[0.5,0.25,0.25,0.5,0.1,0.2,0.3]]))
    u_labels = np.unique(label)  
    for i in u_labels:
        plt.scatter(df_fit[label == i , 0] , df_fit[label == i , 1] , label = i)
    plt.legend()
    plt.show()
    datas = [list(x) for x in zip(df_freq.index.tolist(),label)]
    print(label)
    print(df_freq.index.to_list())
    print(datas)
    df_cluster = pd.DataFrame(data=datas,columns=["id","cluster"])
    print(df_cluster[df_cluster.iloc[:,1]==1].iloc[:,0])
    print(df_final[df_final["tenant_id"].isin( df_cluster[df_cluster.iloc[:,1]==1].iloc[:,0].values)]["property_id"].sample(5))
    with open("./models/cluster_model.pkl","wb") as f:
        pickle.dump(cluster,f)
    with open("./models/cluster_encoder.pkl","wb") as f:
        pickle.dump(encoder,f)
    with open("./models/pca.pkl", "wb") as f:
        pickle.dump(pc,f)
    print(data_to_cluster(df_final,label=label))
if __name__ == "__main__":
    main()
    