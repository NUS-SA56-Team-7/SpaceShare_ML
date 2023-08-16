import types
import numpy as np
import pandas as pd
import random as rd
import json
import os
def seacher_data(n,id):
    data = [[id,rd.randint(0,2),rd.randint(0,3)] for i in range(n)]
    return pd.DataFrame(data=data,columns=["id","property_type","room_type"])
def generate_data(n=None,id=None,df=None):
    if isinstance(df, types.NoneType) == True:
        df= seacher_data(n,id)
    property_type = [0]*3
    room_type = [0]*4
    for x in np.unique(df["property_type"].values):
        property_type[x] = df["property_type"].value_counts()[x]
    for x in np.unique(df["room_type"].values):
        room_type[x] = df["room_type"].value_counts()[x]
    return property_type+room_type
test = generate_data(1,1)
print(test)