import numpy as np
import streamlit as st
import pandas as pd

import seaborn as sns

import joblib

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

advert = pd.read_csv('Advertising.csv')
x = advert.drop('sales', axis=1)
y = advert['sales']
from sklearn.model_selection import train_test_split
train_test_split(x,y,test_size=0.3,random_state=101)
x_train,x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=101)

st.title("Advertisement Budget vs Sales Prediction")


from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
lr_model = LinearRegression()
lr_model.fit(x_train,y_train)
prediction = lr_model.predict(x_test)
from joblib import dump, load
dump(lr_model, 'sales_model.joblib') 
load_model = load('sales_model.joblib')
load_model.predict(x_train)
# lr_model = joblib.load('linear_model.pkl')

# Get user inputs (assume you're using a dataset with TV, Radio, Newspaper)
TV = st.number_input("TV Advertising Budget", min_value=0.0, value=0.0)
radio = st.number_input("Radio Advertising Budget", min_value=0.0, value=0.0)
newspaper = st.number_input("Newspaper Advertising Budget", min_value=0.0, value=0.0)

input_data = np.array([[TV, radio, newspaper]])

if st.button("Predict Sales"):
    prediction = lr_model.predict(input_data)
    st.success(f"Predicted Sales: {prediction[0]:.2f}")







# TV 	= st.sidebar.number_input('TV')
# radio = st.sidebar.number_input('radio')
# newspaper = st.sidebar.number_input('newspaper')

# input_data = ['TV','radio','newspaper', 'sales']
# input_data = np.array(input_data)
# input = input_data.reshape(1,-1)
# st.write('Prediction')
# if st.button('Predict'):
#     prediction = lr_model.predict(input)
#     st.write(prediction)