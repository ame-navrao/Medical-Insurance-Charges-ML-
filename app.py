# -*- coding: utf-8 -*-

from flask import Flask, request
import pandas as pd
import numpy as np
import pickle
import flasgger
from flasgger import Swagger

app = Flask(__name__)
Swagger(app)

pickle_in_regressor = open('Regressor.pkl','rb')

pickle_in_standardScalar = open('StandardScalar.pkl', 'rb')

pickle_in_transformer = open('Transformer.pkl', 'rb')

classifier = pickle.load(pickle_in_regressor)

sc = pickle.load(pickle_in_standardScalar)

encoder = pickle.load(pickle_in_transformer)

@app.route('/')
def welcome():
    return "Medical Cost Personal Dataset Api"


@app.route('/predict')
def predict_note_authentication():
    """Let's Predict the Medical Cost 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: age
        in: query
        type: number
        description: "Enter User Age Here."
        required: true
      - name: sex
        in: query
        type: string
        enum: ["male", "female"]
        description: "Enter User Gender Here."
        required: true
      - name: bmi
        in: query
        type: number
        format: float
        description: "Enter User BMI Here."
        required: true
      - name: children
        in: query
        type: number
        description: "Enter User Children Here."
        required: true
      - name: smoker
        in: query
        type: string
        enum: ["yes", "no"]
        description: "Do you smoke? [yes/no]"
        required: true
      - name: region
        in: query
        type: string
        enum: ["southeast", "southwest", "northwest", "northeast"]
        description: "Enter User Region Here."
        required: true
    responses:
        200:
            description: The output values
        
    """
    age = request.args.get('age')
    sex = request.args.get('sex')
    bmi = request.args.get('bmi')
    children  = request.args.get('children')
    smoker  = request.args.get('smoker')
    region  = request.args.get('region')
    
    X = [
        [
            age,
            sex,
            bmi,
            children,
            smoker,
            region,
            0,
        ]
    ]
    X = encoder.transform(X)
    X = pd.DataFrame(X, columns=encoder.get_feature_names())
    X = X.drop(["charges"], axis=1)
    X = sc.transform(X)
    
    prediction = classifier.predict(X)
    prediction = prediction.astype(str)
    return "The predicted value is " + prediction[0]
       
 
    
if __name__ == '__main__':
    app.run()
    