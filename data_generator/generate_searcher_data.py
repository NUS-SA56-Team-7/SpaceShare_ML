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
    freq = [df["property_type"].value_counts().iloc[x] for x in range(len(df["property_type"].unique()))]+[
            df["room_type"].value_counts().iloc[x] for x in range(len(df["room_type"].unique()))]
    return freq
df = seacher_data(20,1)
df.to_json(path_or_buf="./dataset/test_data.json",orient="records")
df = seacher_data(20,2)
df.to_json(path_or_buf="./dataset/test_data2.json",orient="records")
