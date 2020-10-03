"""
This script runs the application using a development server.
It contains the definition of routes and views for the application.
"""
import pickle
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import requests
import json
from pandas.io.json import json_normalize
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from flask import Flask, render_template, request
app = Flask(__name__)

# Make the WSGI interface available at the top level so wfastcgi can get it.
wsgi_app = app.wsgi_app


@app.route('/solarupload', methods = ["GET", "POST"])
def Uploads():
    """Renders a sample page."""
    if request.method=="POST":
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        solarfile=request.files["file"]
        file.save(os.path.join("Solar", solarfile.filename))
        return render_template("index.html", message = "File Uploaded Successfuly")
    return render_template("index.html", message = "Upload Solar Maintenance File")
    return "File Uploaded!"



@app.route('/windupload', methods = ["GET", "POST"])
def Uploads1():
    """Renders a sample page."""
    if request.method=="POST":
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        windfile=request.files["file"]
        file.save(os.path.join("Wind", windfile.filename))
        return render_template("index.html", message = "File Uploaded Successfuly")
    return render_template("index.html", message = "Upload Wind Maintenance File")
    return "File Uploaded!"


@app.route('/solarprediction', methods = ["GET", "POST"])
def Solar_Prediction():
    """Renders a sample page."""
    model = pickle.load(open('solar_model.pkl','rb'))
    
    # Load DataFrame
    # Solar
    longitude = 142.110216
    latitude = -19.461907

    url = ('https://api.openweathermap.org/data/2.5/onecall?lat=-19.461907&lon=142.110216&units=imperial&appid=43e49f2fb4d17b806dfff389f21f4d27')
    response = requests.get(url)

    weather = response.json()
    dailynorm = json_normalize(weather, 'daily')
    
    df = pd.DataFrame(dailynorm)
    solar_df = df[['dt', 'temp.min', 'temp.max', 'clouds']].copy()
    solar_df['date'] = pd.to_datetime(solar_df['dt'],unit='s')
    solar_df['day'] = solar_df['date'].dt.day
    solar_df['month'] = solar_df['date'].dt.month
    solar_df = solar_df.fillna(0)
    solar_df.rename(columns={'temp.min':'Temp Low',
                          'temp.max':'Temp Hi',
                          'clouds':'Cloud Cover Percentage'}, 
                 inplace=True)
    solar_df = solar_df.drop(['dt','date'], axis = 1)
    Xnew = solar_df.values
    p_predi = model.predict(Xnew)
    p_pred = pd.DataFrame(p_predi)
    p_pred.columns = ['Predicted Power']
    final_solar_df = pd.concat([solar_df, p_pred], axis = 1)
    dictionaryObject = final_solar_df.to_dict()
   
    return dictionaryObject


@app.route('/windprediction', methods = ["GET", "POST"])
def Wind_Prediction():
    """Renders a sample page."""
    model = pickle.load(open('wind_model.pkl','rb'))
    
    # Load DataFrame
    # Wind
    longitude = 53.556563
    latitude = 8.598084

    url = ('https://api.openweathermap.org/data/2.5/onecall?lat=8.598084&lon=53.556563&units=imperial&appid=43e49f2fb4d17b806dfff389f21f4d27')
    response = requests.get(url)

    weather = response.json()
    dailynorm = json_normalize(weather, 'daily')
    
    df = pd.DataFrame(dailynorm)
    wind_df = df[['dt', 'wind_speed', 'wind_deg']].copy()
    wind_df['date'] = pd.to_datetime(wind_df['dt'],unit='s')
    wind_df['day'] = wind_df['date'].dt.day
    wind_df['month'] = wind_df['date'].dt.month
    wind_df = wind_df.fillna(0)
    wind_df.rename(columns={'wind_speed':'wind speed', 'wind_deg':'direction'}, inplace=True)
    wind_df = wind_df.drop(['dt','date'], axis = 1)
    Xnew = wind_df.values
    p_pred = model.predict(Xnew)
    p_pred = pd.DataFrame(p_pred)
    p_pred.columns = ['Predicted Power']
    final_wind_df = pd.concat([wind_df, p_pred], axis = 1)
    dictionaryObject = final_wind_df.to_dict()
   
    return dictionaryObject
    

if __name__ == '__main__':
    import os
    HOST = os.environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(os.environ.get('SERVER_PORT', '5555'))
    except ValueError:
        PORT = 5555
    app.run(HOST, PORT)
