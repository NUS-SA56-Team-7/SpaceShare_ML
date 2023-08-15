import numpy as np
import pandas as pd
import random as rd
def generateFeature(n):
    hdb = n - rd.randint(0,n)
    if(n==hdb):
        return [hdb,0,0]
    n -= hdb
    condo =n - rd.randint(0,n)
    if(n==condo):
        return [hdb,condo,0]
    n -= condo
    return [hdb,condo,n]

def generate_sample_data(n,feature):
    data = []
    for i in range(n):
        features = generateFeature(feature)
        features = generate_room_type(features,feature)
        data.append(features)
    return pd.DataFrame(data,columns=["hdb","condo","landed","single_room","common_room","master_room","whole_unit"])

def generate_room_type(features,n):
    single_room = n - rd.randint(0,n)
    common_room = 0
    master_room = 0
    whole_unit = 0
    if (single_room != n):
        n -= single_room
        common_room = n - rd.randint(0,n)
        if(common_room!=n):
            n -= common_room
            master_room = n - rd.randint(0,n)
            if(master_room != n):
                n -= master_room
                whole_unit = n
    features.append(single_room)
    features.append(common_room)
    features.append(master_room)
    features.append(whole_unit)
    return features
df = generate_sample_data(1000,20)
df.to_csv("./dataset/cluster.csv")