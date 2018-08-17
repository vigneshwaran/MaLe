# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 21:47:56 2018

@author: Vigneshwaran
"""
from sklearn.naive_bayes import GaussianNB
def classify(features_train, labels_train):
    ### import the sklearn module for GaussianNB
    ### create classifier
    ### fit the classifier on the training features and labels
    ### return the fit classifier
    clf = GaussianNB()
    return clf.fit(features_train,labels_train)

    """ ### your code goes here!
    from sklearn.naive_bayes import GaussianNB
    clf=GaussianNB()
    return clf.fit(features_train, labels_train)"""
