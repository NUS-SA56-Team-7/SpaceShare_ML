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
    freq = [df["property_type"].value_counts().iloc[x] for x in range(3)]+[
            df["room_type"].value_counts().iloc[x] for x in range(4)]
    return freq
