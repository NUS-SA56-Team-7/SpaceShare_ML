import pickle as Pickle
import sys
import numpy as np
sys.path.append("./")
import data_generator.generate_searcher_data as gd
import pandas as pd
from flask import Flask, jsonify, request
from waitress import serve
import json
with open("./models/model.pkl",mode="rb") as f:
    model = Pickle.load(f)
with open("./models/scaler.pkl",mode="rb") as f:
    scaler = Pickle.load(f)
app = Flask("__name__")

@app.route("/api/analytics/classify",methods=['Post'])
def clustering():
    try:
        user_data = request.form.get("userData") or request.args.get("userData")
        user_data = json.loads(user_data)
        

    except:
        return "Bad Request",400
    df = pd.json_normalize(user_data)
    if(len(df)!=20):
        return "Bad Request",400
    feature = gd.generate_data(df=df)
    feature = scaler.transform([feature])
    return jsonify(int(model.predict(feature)))
@app.route("/api/analytics/classify/multiple",methods=['Post'])
def multipleclustering():
    try:
        user_data = request.form.get("userData") or request.args.get("userData")
        user_data = json.loads(user_data)
    except:
        return "Bad Request",400
    df = pd.json_normalize(user_data)
    features = [gd.generate_data(df=df[df["id"]==id])for id in np.unique(df["id"].values)]
    features = scaler.transform(features)
    return pd.DataFrame(model.predict(features)).to_json(orient="records")

        

serve(app)