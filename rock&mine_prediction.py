# -*- coding: utf-8 -*-
import numpy
from sklearn import datasets, linear_model
import random
from sklearn.metrics import roc_curve, auc
import pylab as pl
import urllib2
import matplotlib.pyplot as plt


target_url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data")
data = urllib2.urlopen(target_url)

xList = []
labels = []

for line in data:
    row = line.strip().split(",")
    if row[-1] == "M":
        labels.append(1.0)
    else:
        labels.append(0.0)
    row.pop()
    floatRow = [float(num) for num in row]
    xList.append(floatRow)

# devision
indices = range(len(xList))
xListTrain = [xList[i] for i in indices if i%3 != 0]
xListTest = [xList[i] for i in indices if i%3 == 0]
labelsTrain = [labels[i] for i in indices if i%3 != 0]
labelsTest = [labels[i] for i in indices if i%3 == 0]

xTrain = numpy.array(xListTrain);yTrain = numpy.array(labelsTrain)
xTest = numpy.array(xListTest);yTest = numpy.array(labelsTest)

# numpy中的array弥补了Python里面的数组指针冗杂导致的计算能力不足的问题
# 而且可以搞多维数组

# 线性回归
rocksVMinesModel = linear_model.LinearRegression()
rocksVMinesModel.fit(xTrain,yTrain)
# 生成预测
trainingPredictions = rocksVMinesModel.predict(xTrain)
testPredictions = rocksVMinesModel.predict(xTest)

def confusionMatrix(predicted, actual, threshold):
    if len(predicted) != len(actual):return -1 
    tp = 0.0
    fp = 0.0
    fn = 0.0
    tn = 0.0
    for i in range(len(actual)):
        if actual[i] > 0.5:  #True
            if predicted[i] > threshold:  #Passtive
                tp += 1.0
            else:
                tn += 1.0
        else:   #false
            if predicted[i] > threshold:  #negative
                fp += 1.0
            else:
                fn += 1.0
    result =  [(tp+fn)/(tp+tn+fp+fn),threshold]
    return result
    
thress = 0.0
ylabel = []
xlabel = []
# generate confusion matrix for predictions on training set
while thress <= 1:
    correct_percentage = confusionMatrix(testPredictions, yTest, thress)
    xlabel.append(correct_percentage[1])
    ylabel.append(correct_percentage[0])
    thress += 0.01
plt.plot(xlabel,ylabel,label="$sin(x)$",color="red",linewidth=2)  
plt.show()








