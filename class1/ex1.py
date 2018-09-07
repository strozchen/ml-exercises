# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 16:56:52 2018
Linear regression with one variable
@author: admin
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#compute cost funtion
def computeCost(X,y,theta):
    inner=np.power(((X*theta.T)-y),2) 
    J=np.sum(inner)
    J_m=J/(2*len(X))
    return J_m

#compute gradient funciton
def gradientDescent(X,y,theta,alpha,iters):
    temp=np.matrix(np.zeros(theta.shape))
    parameters=int(theta.ravel().shape[1])
    cost=np.zeros(iters)
    
    for i in range(iters):
        error=(X*theta.T)-y
        for j in range(parameters):
            term=np.multiply(error,X[:,j])
            temp[0,j]=theta[0,j]-((alpha/len(X))*np.sum(term))
            
        theta=temp
        cost[i]=computeCost(X,y,theta)
    return theta,cost

path='ex1data1.txt'
data=pd.read_csv(path,header=None,names=['Population','Profit'])
#print(data.head())
#print(data.describe())

data.plot(kind='scatter',x='Population',y='Profit',figsize=(12,8))
plt.show()

data.insert(0,'Ones',1)

cols=data.shape[1]
X=data.iloc[:,0:cols-1]
y=data.iloc[:,cols-1:cols]

X=np.matrix(X.values)
y=np.matrix(y.values)
theta=np.matrix(np.array([0.01,0.01]))

#rs=computeCost(X,y,theta)
alpha=0.01
iters=1000
g,cost=gradientDescent(X,y,theta,alpha,iters)
#print(g)

x = np.linspace(data.Population.min(), data.Population.max(), 100)
f=g[0,0]+(g[0,1]*x)
fig,ax=plt.subplots(figsize=(12,8))

ax.plot(x,f,'r',label='Prediction')
ax.scatter(data.Population,data.Profit,label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()

