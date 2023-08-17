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
df_history = pd.read_csv("./dataset/mock_data.csv",index_col=0)
df_property = pd.read_csv("./dataset/mock_property.csv",index_col=0)
#for loading of df from backend
#response = requests.post()
#property_json = json.loads(response)
#property_df = pd.json_normalize(property_json)
df_cluster = pd.read_csv("./dataset/cluster_data.csv",index_col=0)
app = Flask("__name__")

@app.route("/api/analytics/classify",methods=['Post'])
def clustering():
   refresh()
   id = request.form.get("id") or request.args.get("id")
   df = df_history[df_history["tenant_id"]==int(id)]
   df_list = gd.df_to_freq(df,id)
   df_list.pop(0)
   data = scalar.transform([df_list])
   cluster = model.predict(data)
   sim_cluster = df_cluster[df_cluster["cluster"]==cluster[0]]['id'].to_list()
   property_reco = df_history[df_history["tenant_id"].isin(sim_cluster)]["property_id"]
   return property_reco.sample(5).to_json()
@app.route("/api/analytics/load_today_cluster",methods =['Get'])
def cache():
    refresh()
    df_history[["property_type","room_type"]] = encoder.transform(df_history[["property_type","room_type"]])
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
    return "Success", 200
def refresh():
    print("TEST")
    #response = requests.post()
    #property_json = json.loads(response.json())
    #df_property = pd.json_normalize(property_json)
    #response.close()
    #response = requests.post()
    #history_json = json.loads(response.json())
    #df_history = pd.json_normalize(property_json)
    #response.close()
    #to reload the memory of new df
    pass
@app.route("/api/analytics/retrain")
def retrain():
    refresh()
    #model.fit(new data)
    #save model in pickle
    pass
scheduler = BackgroundScheduler()
scheduler.add_job(func=refresh,trigger="interval",hours=1)
scheduler.add_job(func=cache,trigger="interval",hours=1)
scheduler.start()
serve(app)