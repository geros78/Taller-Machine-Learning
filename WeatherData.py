# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 23:14:32 2022

@author: alejo
"""


import numpy as np
import pandas as pd

url = './Weather/weatherAUS.csv'
data = pd.read_csv(url)




data.RainToday.replace(['No', 'Yes'], [0, 1], inplace=True)
data.RainTomorrow.replace(['No', 'Yes'], [0, 1], inplace=True)

data.WindSpeed9am.replace(np.nan, 120, inplace=True)
rangos = [ 1, 26, 52 ,78, 94, 110, 130]
nombres = ['1', '2', '3', '4', '5', '6']
data.WindSpeed9am = pd.cut(data.WindSpeed9am, rangos, labels=nombres)

data.WindSpeed3pm.replace(np.nan, 120, inplace=True)
rangos = [ 1, 17, 34, 52, 69, 87]
nombres = ['1', '2', '3', '4', '5']
data.WindSpeed3pm = pd.cut(data.WindSpeed3pm, rangos, labels=nombres)

data.Humidity9am.replace(np.nan, 120, inplace=True)
rangos = [ 0, 20, 40, 60, 80, 100]
nombres = ['1', '2', '3', '4', '5']
data.Humidity9am = pd.cut(data.Humidity9am, rangos, labels=nombres)

data.Humidity3pm.replace(np.nan, 120, inplace=True)
rangos = [ 0, 20, 40, 60, 80, 100]
nombres = ['1', '2', '3', '4', '5']
data.Humidity3pm = pd.cut(data.Humidity3pm, rangos, labels=nombres)

data.Pressure9am.replace(np.nan, 120, inplace=True)
rangos = [ 0, 20, 40, 60, 80, 100]
nombres = ['1', '2', '3', '4', '5']
data.Pressure9am = pd.cut(data.Pressure9am, rangos, labels=nombres)



data.drop(['Date','Location','Evaporation','Sunshine'], axis= 1, inplace = True)

'''data.dropna(axis=0,how='any', inplace=True)'''
















