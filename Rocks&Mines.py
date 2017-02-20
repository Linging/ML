# -*- coding: utf-8 -*-
import pandas as pd
from pandas import DataFrame
from random import uniform
import matplotlib.pyplot as plot

target = ("https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data")

data = pd.read_csv(target, header=None, prefix="V")
#for i in range(208):
#    if data.iat[i,60] == 'M':
#        pcolor = "red"
#    else:
#        pcolor = "blue"
#    dataRow = data.iloc[i,0:60]
#    dataRow.plot(color = pcolor)
#    
#plot.xlabel("Index")
#plot.ylabel("value")
#plot.show()
#由以上code绘制的平行坐标图，Mines&Rocks在12，21，36处具有分离的趋势


#dataRow2 = data.iloc[11,0:60]
#dataRow3 = data.iloc[20,0:60]
#
#plot.scatter(dataRow2, dataRow3)
#
#plot.xlabel("12nd Attribute")
#plot.ylabel("21rd Attribute")
#plot.show()
#取两个属性作为x,y轴，绘制属性交会图（散点），直接体现了属性之间的相关性
#呈线性-->强相关  呈球形-->弱相关


#target = []
#for i in range(208):
#    if data.iat[i,60] == "M":
#        target.append(1.0 + uniform(-0.1,0.1))
#    else:
#        target.append(uniform(-0.1,0.1))
#dataRow = data.iloc[0:208,35]
#plot.scatter(dataRow, target, alpha=0.5, s=120)
#热图
corMat = DataFrame(data.corr())
##alpha 透明度
#plot.xlabel("36ed index")
#plot.ylabel("tag value")
#plot.show()
#标签与属性的交会图


plot.pcolor(corMat)
plot.show()








