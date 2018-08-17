# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 19:36:00 2018

@author: Vigneshwaran
"""

import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

file=pd.read_csv(r'C:\Users\Vigneshwaraen\Desktop\bin\data.csv')
file.columns=['first','sec','third','four','name']
first=file[['first']].values.tolist()
sec=file[['four']].values.tolist()
data=linear_model.LinearRegression()
data.fit(first,sec)
plt.scatter(first,sec,c='red')
plt.plot(first,data.predict(first))
plt.show()

