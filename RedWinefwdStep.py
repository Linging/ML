import numpy
from math import sqrt
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
import urllib2

target_url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv")
data = urllib2.urlopen(target_url)

xlist = []
labels = []

for line in data:
    row = line.strip().split(";")
    labels.append(float(row[-1]))
    row.pop()
    row.pop(0)
    floatRow = [float(num) for num in row]
    xlist.append(floatRow)
labels.pop(0)

# division
indices = range(len(xlist))
xlistTest = [xlist[i] for i in indices if i%3 == 0]    
xlistTrain = [xlist[i] for i in indices if i%3 != 0]
labelTest = [xlist[i] for i in indices if i%3 == 0]
labelTrain = [xlist[i] for i in indices if i%3 != 0]

xTrain = numpy.array(xlistTrain)
xTest = numpy.array(xlistTest)
yTrain = numpy.array(labelTrain)
yTest = numpy.array(labelTest)

alphalist = [0.1**i for i in [0,1,2,3,4,5,6]]
rmsError = []

for alpha in alphalist:
    wineRidgeModel = linear_model.Ridge(alpha=alpha)
    wineRidgeModel.fit(xTrain,yTrain)
    rmsError.append(numpy.linalg.norm((yTest-wineRidgeModel.predict(xTest)),2)/sqrt(len(yTest)))
    
print("RMS Error      |     alpha")
for i in range(len(rmsError)):
    print(rmsError[i], alphalist[i])

# plot curve of out-of-sample error versus alpha
x = range(len(rmsError))