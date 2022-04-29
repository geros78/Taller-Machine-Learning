# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 18:25:46 2022

@author: alejo
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Bank data

url = './BankMarketing/bank-full.csv'
data = pd.read_csv(url)

# Limpiar la data

data.job.replace(['blue-collar','management','technician','admin.','services','retired',
'self-employed', 'entrepreneur','unemployed','housemaid','student','unknown'], 
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], inplace=True)

data.education.replace(['unknown', 'primary', 'secondary', 'tertiary'], [0, 1, 2, 3], inplace=True)
data.default.replace(['no', 'yes'], [0, 1], inplace=True)
data.marital.replace(['married', 'single', 'divorced'], [0, 1, 2], inplace=True)
data.housing.replace(['no', 'yes'], [0, 1], inplace=True)
data.loan.replace(['no', 'yes'], [0, 1], inplace=True)
data.poutcome.replace(['unknown', 'failure', 'other', 'success'], [0, 1, 2, 3], inplace=True)
data.y.replace(['no', 'yes'], [0, 1], inplace=True)

#Datos en rango

rangosAge = [20, 30, 40, 50, 70, 100]
nombresAge = ['1', '2', '3', '4', '5']
data.age = pd.cut(data.age, rangosAge, labels=nombresAge)


rangoCam = [10, 20, 30, 40, 50, 70]
nombreCam = ['1', '2', '3', '4', '5']
data.campaign = pd.cut(data.campaign, rangoCam, labels=nombreCam)

#Datos desechados

data.drop(['balance','contact','day','month','duration',
           'pdays','previous','campaign'], axis= 1, inplace = True)

data.dropna(axis=0,how='any', inplace=True)

#Dividir la data

data_train = data[:22605]
data_test = data[22605:]


x = np.array(data_train.drop(['y'], 1))
y = np.array(data_train.y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.1)

x_test_out = np.array(data_test.drop(['y'], 1))
y_test_out = np.array(data_test.y)






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






















