import pickle
from django.shortcuts import render
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()

app = Flask(__name__, template_folder='template')

## Load the model
regmodel=pickle.load(open('regmodel.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict_api',methods=['Post'])
def predictr_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    ss.fit(np.array(list(data.values())).reshape(1,-1))
    new_data=ss.transform(np.array(list(data.values())).reshape(1,-1))
    output=regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['Post'])
def predict():
    data=[float(x)for x in request.form.values()]
    final_input=ss.fit_transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=regmodel.predict(final_input)[0]
    return render_template("home.html",prediction_text="The House Price Pridiction is {}".format(output))
if __name__=="__main__":
    app.run(debug=True)