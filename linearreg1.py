# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 23:42:12 2018

@author: Vigneshwaran
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# read the Houses dataset CSV file
housing_data = pd.read_csv(r"C:\Users\Vigneshwaraen\Desktop\RealEstate.csv", sep=",")
#plt.hist(housing_data)
# only get the Size and the Price features
Xs = housing_data[['Size']]
Ys = housing_data[['Price']]

# get some statistics
max_size = np.max(Xs)
min_size = np.min(Xs)
max_price = np.max(Ys)
min_price = np.min(Ys)

# Normalize the input features
Xs = (Xs - min_size) / (max_size - min_size)
Ys = (Ys - min_price) / (max_price - min_price)

plt.plot(Xs,Ys,"red",linestyle='dashed')
plt.legend()
plt.show()



for Xs_batch, Ys_batch in next_batch(Xs, Ys, batch_size=128):

    # linearly combine input and weights
    Y_pred = W0 + np.dot(W1, Xs_batch)

    # calculate the SSE between predicted and true values
    err = 1/2 * sum((Ys_batch-Y_pred)**2)

    # calculate the gradients with respect to W0 and W1
    DW0 = - (Y_pred-Ys_batch)
    DW1 = - (Xs_batch * (Y_pred - Ys_batch))

    # update W0 and W1 in the opposite direction to the gradient
    W0 = W0 + lr * sum(DW0)
    W1 = W1 + lr * sum(DW1)
    
plt.plot(Xs,Ys,"red",linestyle='dashed')
plt.legend()
plt.show()