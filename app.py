from flask import Flask, render_template, url_for, request, jsonify
import joblib
import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as ptly
import cufflinks as cf
import pickle
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from  sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
# import the regressor
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation,Dense
from sklearn.metrics import mean_squared_error,r2_score



app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')


@app.route("/result",methods=['POST','GET'])
def result():
    temp = float(request.form['temp'])
    rain = float(request.form['rain'])
    snow = float(request.form['snow'])
    clouds = int(request.form['clouds'])
    weather_main = request.form['weather_main']
    weather_description = request.form['weather_description']
    month = int(request.form['month'])
    time = int(request.form['time'])


    x = np.array([temp, rain, snow, clouds, weather_main, weather_description, month, time]).reshape(1, -1)

    print(x)

    ll = x.tolist()
    print(ll)

    with open("encoder.pkl", "rb") as f:
        encoder = pickle.load(f)

    encoded_new_data = encoder.transform(np.array(ll[0][4]).reshape(-1, 1))
    ll[0] += encoded_new_data.toarray().tolist()[0]

    with open("encoder_w.pkl", "rb") as f:
        encoder_w = pickle.load(f)

    encoded_new_data_w = encoder_w.transform(np.array(ll[0][5]).reshape(-1, 1))
    ll[0] += encoded_new_data_w.toarray().tolist()[0]
    with open("l2_normalized.pkl", "rb") as f:
        normalization_transform = pickle.load(f)

    ll[0].remove(weather_main)
    ll[0].remove(weather_description)

    normalized_new_data = normalize(ll, norm='l2')
    from tensorflow.keras.models import load_model
    model = load_model(r"C:\Users\udays\Desktop\Data Science\projects\metro\model.h5")
    prediction = model.predict(normalized_new_data)
    prediction = float(prediction)
    print(prediction)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    y_pred_l2_n_hypo_a = scaler.inverse_transform([prediction])
    print(y_pred_l2_n_hypo_a)

    return render_template('result.html', res= y_pred_l2_n_hypo_a )





if __name__=="__main__":
    app.run(debug=True, port=5298)