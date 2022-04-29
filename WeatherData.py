# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 23:14:32 2022

@author: alejo
"""


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

url = './Weather/weatherAUS.csv'
data = pd.read_csv(url)




data.RainToday.replace(['No', 'Yes'], [0, 1], inplace=True)
data.RainTomorrow.replace(['No', 'Yes'], [0, 1], inplace=True)


#Limpieza de datos en Rangos 

data.WindSpeed9am.replace(np.nan, 14, inplace=True)
rangos = [ 1, 26, 52 ,78, 94, 110, 130]
nombres = ['1', '2', '3', '4', '5', '6']
data.WindSpeed9am = pd.cut(data.WindSpeed9am, rangos, labels=nombres)

data.WindSpeed3pm.replace(np.nan, 19, inplace=True)
rangos = [ 1, 17, 34, 52, 69, 87]
nombres = ['1', '2', '3', '4', '5']
data.WindSpeed3pm = pd.cut(data.WindSpeed3pm, rangos, labels=nombres)

data.Humidity9am.replace(np.nan, 69, inplace=True)
rangos = [ 0, 20, 40, 60, 80, 100]
nombres = ['1', '2', '3', '4', '5']
data.Humidity9am = pd.cut(data.Humidity3pm, rangos, labels=nombres)

rangos = [ 0, 20, 40, 60, 80, 100]
nombres = ['1', '2', '3', '4', '5']
data.Humidity3pm = pd.cut(data.Humidity3pm, rangos, labels=nombres)

rangos = [ 980, 994, 1008, 1022, 1036, 1050]
nombres = ['1', '2', '3', '4', '5']
data.Pressure9am = pd.cut(data.Pressure9am, rangos, labels=nombres)

rangos = [ 970, 984, 998, 1012, 1026, 1040]
nombres = ['1', '2', '3', '4', '5']
data.Pressure3pm = pd.cut(data.Pressure3pm, rangos, labels=nombres)

data.Cloud9am.replace(np.nan, 4, inplace=True)
rangos = [ 0, 1, 2, 3, 4, 5, 6, 7, 9]
nombres = ['1', '2', '3', '4', '5', '6', '7', '8']
data.Cloud9am = pd.cut(data.Cloud9am, rangos, labels=nombres)

data.Cloud3pm.replace(np.nan, 5, inplace=True)
rangos = [ 0, 1, 2, 3, 4, 5, 6, 7, 9]
nombres = ['1', '2', '3', '4', '5', '6', '7', '8']
data.Cloud3pm = pd.cut(data.Cloud3pm, rangos, labels=nombres)

rangos = [ -8, 0, 10, 20, 30, 42]
nombres = ['1', '2', '3', '4', '5']
data.Temp9am = pd.cut(data.Temp9am, rangos, labels=nombres)

rangos = [ -6, 5, 15, 25, 40, 50]
nombres = ['1', '2', '3', '4', '5']
data.Temp3pm = pd.cut(data.Temp3pm, rangos, labels=nombres)

rangos = [ -8, 0, 10, 20, 35]
nombres = ['1', '2', '3', '4']
data.MinTemp = pd.cut(data.MinTemp, rangos, labels=nombres)

rangos = [ -5, 10, 20, 30, 40, 50]
nombres = ['1', '2', '3', '4', '5']
data.MaxTemp = pd.cut(data.MaxTemp, rangos, labels=nombres)

data.WindGustSpeed.replace(np.nan, 41, inplace=True)
rangos = [ 7, 30, 50, 70, 90, 110, 135]
nombres = ['1', '2', '3', '4', '5', '6']
data.WindGustSpeed = pd.cut(data.WindGustSpeed, rangos, labels=nombres)

data.drop(['Date','Location','Evaporation','Sunshine', 'RISK_MM', 'WindGustDir'
           ,'WindDir9am', 'WindDir3pm', 'Rainfall'], axis= 1, inplace = True)

data.dropna(axis=0,how='any', inplace=True)








#Dividir la data

data_train = data[:20000]
data_test = data[20000:]


x = np.array(data_train.drop(['RainTomorrow'], 1))
y = np.array(data_train.RainTomorrow)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.1)

x_test_out = np.array(data_test.drop(['RainTomorrow'], 1))
y_test_out = np.array(data_test.RainTomorrow)






# MODELO DE REGRESION LOGICA
logreg = LogisticRegression(solver= 'lbfgs', max_iter= 7600)

#Entrenamiento del Modelo
logreg.fit(x_train, y_train)

#Metricas del modelo

print('-'*60)
print('Regresion Logistica')

#Presicion del test de Entrenamiento de entrenamiento

print(f'accuracy de Entrenamiento de Entrenamiento: {logreg.score (x_train, y_train)}')

#Presicion del test de Entrenamiento

print(f'accuracy de test Entrenamiento: {logreg.score (x_test, y_test)}')

#Presicion del Validacio

print(f'accuracy de Validacion: {logreg.score (x_test_out, y_test_out)}')




# MAQUINA DE SOPORTE VECTORIAL

# Seleccionar un Modelo
svc = SVC(gamma='auto')

#Entreno el modelo
svc.fit(x_train, y_train)

#Metricas del modelo

print('-'*60)
print('Maquina de soporte vectorial')

#accuracy de test de Entrenamiento de entrenamiento

print(f'accuracy de Entrenamiento de Entrenamiento: {svc.score (x_train, y_train)}')

#accuracy de test de Entrenamiento

print(f'accuracy de test Entrenamiento: {svc.score (x_test, y_test)}')

#accuracy de Validacio

print(f'accuracy de Validacion: {svc.score (x_test_out, y_test_out)}')





# ARBOL DE DESCISION

# Seleccionar un Modelo
arbol = DecisionTreeClassifier()

#Entreno el modelo
arbol.fit(x_train, y_train)

#Metricas del modelo

print('-'*60)
print('Decision Tree')

#accuracy de test de Entrenamiento de entrenamiento

print(f'accuracy de Entrenamiento de Entrenamiento: {arbol.score (x_train, y_train)}')

#accuracy de test de Entrenamiento

print(f'accuracy de test Entrenamiento: {arbol.score (x_test, y_test)}')

#accuracy de Validacio

print(f'accuracy de Validacion: {arbol.score (x_test_out, y_test_out)}')











