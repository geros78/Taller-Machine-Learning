# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 22:10:12 2022

@author: alejo
"""

import numpy as np
import pandas as pd

url = './DiabetesDatabase/diabetes.csv'
data = pd.read_csv(url)

# Limpiar la data

data.Glucose.replace(np.nan, 120, inplace=True)
rangosGlu = [ 70, 100 ,120, 150, 170, 200]
nombresGlu = ['1', '2', '3', '4', '5']
data.Glucose = pd.cut(data.Glucose, rangosGlu, labels=nombresGlu)

rangosAge = [ 20, 30, 40, 50, 70, 100]
nombresAge = ['1', '2', '3', '4', '5']
data.Age = pd.cut(data.Age, rangosAge, labels=nombresAge)

data.BMI.replace(np.nan, 32, inplace=True)
rangosBMI = [ 10, 20, 30, 40, 50, 70]
nombresBMI = ['1', '2', '3', '4', '5']
data.BMI = pd.cut(data.BMI, rangosBMI, labels=nombresBMI)

rangosDia = [ 0.05, 0.25, 0.50, 1, 1.50, 2.50]
nombresDia = ['1', '2', '3', '4', '5']
data.DiabetesPedigreeFunction = pd.cut(data.DiabetesPedigreeFunction, rangosDia, labels=nombresDia)

























