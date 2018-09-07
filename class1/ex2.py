# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 10:33:03 2018
Linear regression with multiple variables
@author: admin
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
df=pd.read_csv("ex1data2.txt", header=None,names=['size','roomnums','price'])
df=(df-df.mean())/df.std()

X=df.iloc[:,0:1]
Y=df.iloc[:,1:2]
H=df.iloc[:,2:3]
s=np.ones(X.shape)
'''
#显示读到的数据
fig=plt.figure()
ax = fig.add_subplot(111, projection='3d')
X=np.array(X)
Y=np.array(Y)
H=np.array(H)
ax.scatter(X, Y, H, c='b', marker='o')
plt.show()
'''

def computeCost(X,y,theta):
    inner=np.power(((X*theta.T)-y),2) 
    J=np.sum(inner)
    J_m=J/(2*len(X))
    return J_m

#X:输入
#y：输出
#theta：参数
#alpha：梯度下降系数
#iters：迭代次数
def gradientDescent(X,y,theta,alpha,iters):
    temp=np.matrix(np.zeros(theta.shape))
    parameters=int(theta.ravel().shape[1])
    cost=np.zeros(iters)
    
    for i in range(iters):
        error=(X*theta.T)-y
        for j in range(parameters):
            #此处为点乘，每个同位置元素相乘，故当X不为矩阵时，计算出的矩阵维度不对
            term=np.multiply(error,X[:,j])
            temp[0,j]=theta[0,j]-((alpha/len(X))*np.sum(term))
            
        theta=temp
        cost[i]=computeCost(X,y,theta)
    return theta,cost

#正规方法计算theta=(X.T*X)^-1*X.T*y
def normalEqn(X,y):
    theta=np.linalg.inv(X.T@X)@X.T@y
    return theta


Xs=np.column_stack((s,X,Y))
Xs=np.matrix(Xs)#将array转为矩阵形式，避免后面迭代时候点乘出错
theta=np.matrix(np.array([0.01,0.01,0.01]))
alpha=0.02
iters=800
#迭代计算
g,cost=gradientDescent(Xs,H,theta,alpha,iters)
#计算单个代价函数
#j=computeCost(Xs,H,g)
#print(j)
#正规方程计算
g2=normalEqn(Xs,H)

#误差曲线
#fig, ax = plt.subplots()
#ax.plot(np.arange(iters), cost, 'r')
#ax.set_xlabel('Iterations')
#ax.set_ylabel('Cost')
#ax.set_title('Error vs. Training Epoch')
#plt.show()

Hs=Xs*g.T
Hs2=Xs*g2

fig=plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax = fig.add_subplot(111)
X=np.array(X)
Y=np.array(Y)
H=np.array(H)
Hs=np.array(Hs)
ax.scatter(X, Y, H, c='b', marker='o')
ax.scatter(X, Y, Hs, c='r', marker='o')
ax.scatter(X, Y, Hs2, c='y', marker='o')
plt.show()



