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
relativPath = 'data'

url = 'https://raw.githubusercontent.com/diogocortiz/Crash-Course-IA/master/RegressaoLinear/' + filename
urllib.request.urlretrieve(url, relativPath+'/'+filename)

# loading the dataset for a Dataframe (Pandas)

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

# Split the dataset into training data and test data
# in this case we will use scikitlearn's train_test_split
motores_treino, motores_test, co2_treino, co2_teste = train_test_split(motores, co2, test_size=0.2, random_state=42)
print(type(motores_treino))

# display the correlation between the features of the training dataset
plt.scatter(motores_treino, co2_treino, color='blue')
plt.xlabel("Motor")
plt.ylabel("Emissão de CO2")
plt.show()

# Let's train the linear regression model

# create a linear regression type model
modelo =  linear_model.LinearRegression()

# training the model using the test dataset
# to find the value of A and B (Y = A + B.X)

modelo.fit(motores_treino, co2_treino)

# display the coefficients (A and B)
print('(A) Intercepto: ', modelo.intercept_)
print('(B) Inclinação: ', modelo.coef_)

# display our regression line in the training dataset
plt.scatter(motores_treino, co2_treino, color='blue')
plt.plot(motores_treino, modelo.coef_[0][0]*motores_treino + modelo.intercept_[0], '-r')
plt.ylabel("Emissão de C02")
plt.xlabel("Motores")
plt.show()

# let's run our model in the test dataset

# make predictions using the model and test base
predicoesCo2 = modelo.predict(motores_test)

# display our regression line in the test dataset
plt.scatter(motores_test, co2_teste, color='blue')
plt.plot(motores_test, modelo.coef_[0][0]*motores_test + modelo.intercept_[0], '-r')
plt.ylabel("Emissão de C02")
plt.xlabel("Motores")
plt.show()

# evaluate the model
# show the merics
print("Soma dos Erros ao Quadrado (SSE): %.2f " % np.sum((predicoesCo2 - co2_teste)**2))
print("Erro Quadrático Médio (MSE): %.2f" % mean_squared_error(co2_teste, predicoesCo2))
print("Erro Médio Absoluto (MAE): %.2f" % mean_absolute_error(co2_teste, predicoesCo2))
print ("Raiz do Erro Quadrático Médio (RMSE): %.2f " % sqrt(mean_squared_error(co2_teste, predicoesCo2)))
print("R2-score: %.2f" % r2_score(predicoesCo2 , co2_teste) )