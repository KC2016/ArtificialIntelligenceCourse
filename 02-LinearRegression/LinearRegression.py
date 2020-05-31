'''Crash Course IA - Lesson 2 - Linear Regression
This is the first practical exercise of the AI Crash Course, 
an initiative to adapt the AI discipline of TIDD / PUC-SP to an online version 
that is accessible to everyone.
In this tutorial, we will train a linear regression model for predicting continuous values. 
Classes are available on YouTube: https://youtu.be/Ze-Q6ZNWpco'''
from matplotlib import pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split
from math import sqrt



# download the dataset
import urllib.request
    
filename = 'FuelConsumptionCo2.csv'
relativPath = '02-LinearRegression/data'

url = 'https://raw.githubusercontent.com/diogocortiz/Crash-Course-IA/master/RegressaoLinear/' + filename
urllib.request.urlretrieve(url, relativPath+'/'+filename)

#loading the dataset for a Dataframe (Pandas)

# create a dataset called 'df' that will receive csv data
df = pd.read_csv(relativPath+'/'+filename)

# display the dataframe structure
print(df.head())

# display the Dataset summary
print(df.describe())

# select only the Engine and CO2 features
motores =  df[['ENGINESIZE']]
co2 = df[['CO2EMISSIONS']]
print(motores.head())