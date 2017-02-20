# -*- coding: utf-8 -*-
import pandas as pd
from pandas import DataFrame
import matplotlib as pl
import matplotlib.pyplot as plot
from pylab import *
from math import exp 


target_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
abalone = pd.read_csv(target_url, header=None, prefix="V")
abalone.columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']

summary = abalone.describe()

#abaloneNormalized = abalone.iloc[:,1:9]
#for i in range(8):
#    mean = summary.iloc[1,i]
#    std = summary.iloc[2,i]
#    abaloneNormalized.iloc[:,i:i+1] = (abaloneNormalized.iloc[:,i:(i+1)] - mean) / std
#归一化(normalized)     Xi - u
#                  X = --------
#                       DX^0.5

#array = abaloneNormalized.values
#boxplot(array)
#plot.xlabel = "attribute index"
#show()

minRing = summary.iloc[3,7]
maxRing = summary.iloc[7,7]
meanRing = summary.iloc[1,7]
stdRing = summary.iloc[2,7]

rows = len(abalone.index) 
for i in range(rows):
    dataRow = abalone.iloc[i,1:8]
    normTarget = (abalone.iloc[i,8] - meanRing) / stdRing
    labelcolor = 1.0 / (1.0 + exp(-normTarget))
    dataRow.plot(color=pl.cm.RdYlBu(labelcolor), alpha=0.5)
show()    
#
