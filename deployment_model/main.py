import pickle as Pickle
import sys
import requests
import numpy as np
sys.path.append("./")
import data_generator.generate_searcher_data as gd
import pandas as pd
from flask import Flask, jsonify, request
from waitress import serve
import json
import time
import atexit
from apscheduler.schedulers.background import BackgroundScheduler

with open("./models/cluster_model.pkl",mode="rb") as f:
    model = Pickle.load(f)
with open("./models/cluster_encoder.pkl",mode="rb") as f:
    encoder = Pickle.load(f)
with open("./models/pca.pkl",mode= "rb") as f:
    scalar = Pickle.load(f)
#production replace them with empty df
#for loading of df from backend
#response = requests.post()
#property_json = json.loads(response.content)
#property_df = pd.json_normalize(property_json)
#can just call refresh instead
df_cluster = pd.read_csv("./dataset/cluster_data.csv",index_col=0)
app = Flask("__name__")

@app.route("/api/analytics/classify",methods=['Post'])
def clustering():
   id = request.form.get("id") or request.args.get("id")
   #it will be better if we can just get all the information we need
   response = requests.get("http://current.stephenphyo.com:8000/api/tenant/{}/recents".format(id))
   df = pd.json_normalize(response.json())
   df = df.rename(columns={"propertyType":"property_type","roomType":"room_type"})
   df = df[['property_type','room_type']]
   #df = df_history[df_history["tenant_id"]==int(id)]
   df_list = gd.df_to_freq(df,id)
   df_list.pop(0)
   data = scalar.transform([df_list])
   cluster = model.predict(data)
   sim_cluster = df_cluster[df_cluster["cluster"]==cluster[0]]['id'].to_list()
   property_reco = df_history[df_history["tenant_id"].isin(sim_cluster)]["id"]
   return property_reco.sample(5).to_json()
@app.route("/api/analytics/load_today_cluster",methods =['Get'])
def cache():
    global df_cluster
    data = []
    for i in np.unique(df_history['tenant_id']):
        data.append(gd.df_to_freq(df_history[df_history["tenant_id"]==i],i))
    df = pd.DataFrame(data=data,columns=["tenant_id","CONDOMINIUM","HDB","LANDED","COMMON","MASTER","SINGLE","WHOLE_UNIT"]
)
    df = df.set_index("tenant_id")
    features = scalar.transform(df.values)
    label = model.predict(features)
    df = gd.data_to_cluster(df_history,label)
    df.to_csv("./dataset/cluster_data.csv")
    df_cluster = df
    return "Success", 200
def refresh():
    global df_history
    global df_property
    response = requests.get("http://current.stephenphyo.com:8000/api/property")
    df_property = pd.json_normalize(response.json()['data']['content'])
    response.close()
    response = requests.get("http://current.stephenphyo.com:8000/api/tenant")
    df_tenant = pd.json_normalize(response.json())
    response.close()
    dfs = []
    for id in df_tenant['id']:
        response = requests.get("http://current.stephenphyo.com:8000/api/tenant/{}/recents".format(id))
        data = response.json()
        for ele in data:
            ele["tenant_id"]=id
        df = pd.json_normalize(data)
        dfs.append(df)
    df_history = pd.concat(dfs)[['tenant_id','id','propertyType','roomType']]
    df_history=df_history.rename(columns={"propertyType":"property_type","roomType":"room_type"})
    df_history[['property_type','room_type']] = encoder.transform(df_history[['property_type','room_type']])
    
    
    #response = requests.post()
    #df_history = pd.json_normalize(response.json())
    #response.close()

    pass
refresh()
cache()
scheduler = BackgroundScheduler()
scheduler.add_job(func=refresh,trigger="interval",hours=1)
scheduler.add_job(func=cache,trigger="interval",hours=1)
#scheduler.start()
serve(app)