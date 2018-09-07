# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 14:00:33 2018

@author: admin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import seaborn as sns
#from sklearn.metrics import classification_report
def sigmoid(z):
    return 1/(1+np.exp(-z))

def cost(theta,x,y,lamda):
    X=np.matrix(x)
    Y=np.matrix(y)
    theta=np.matrix(theta)
    first=np.log(sigmoid(X*theta.T))   
    first=np.multiply(first,Y)
    second=np.multiply((1-Y),np.log(1-sigmoid(X*theta.T)))  
    s1=-1*first-second
    j1=s1.sum()/len(X)
    j2=(np.multiply(theta,theta).sum())*lamda/(2*len(X))
    return j1+j2

def gradDec(theta,x,y,lamda):
    theta=np.matrix(theta)
    X=np.matrix(x)
    Y=np.matrix(y)
    grad=np.zeros(theta.ravel().shape[1])
    er=np.multiply(sigmoid(X*theta.T)-Y,X) 
#    er = sigmoid(X * theta.T) - Y
    for i in range(theta.ravel().shape[1]):
        tmp=er[:,i:i+1]
#        tmp = np.multiply(er, X[:,i])
        if i!=0:
            grad[i]=tmp.sum()/er.shape[0]+lamda*theta[:,i]/er.shape[0]
        else:
            grad[i]=tmp.sum()/er.shape[0]
    return grad
#计算预测结果
def predict(theta,x):
    X=np.matrix(x)
    predictrs=sigmoid(X*theta.T)
    return [1 if x>=0.5 else 0 for x in predictrs]

def feature_mapping(x, y, power, as_ndarray=False):
#     """return mapped features as ndarray or dataframe"""
    # data = {}
    # # inclusive
    # for i in np.arange(power + 1):
    #     for p in np.arange(i + 1):
    #         data["f{}{}".format(i - p, p)] = np.power(x, i - p) * np.power(y, p)

    data = {"f{}{}".format(i - p, p): np.power(x, i - p) * np.power(y, p)
                for i in np.arange(power + 1)
                for p in np.arange(i + 1)
            }

    if as_ndarray:
        return pd.DataFrame(data).as_matrix()
    else:
        return pd.DataFrame(data)
#寻找决策边界函数
def find_decision_boundary(density, power, theta, threshhold):
    t1 = np.linspace(-1, 1.5, density)
    t2 = np.linspace(-1, 1.5, density)

    cordinates = [(x, y) for x in t1 for y in t2]
    x_cord, y_cord = zip(*cordinates)
    mapped_cord = feature_mapping(x_cord, y_cord, power)  # this is a dataframe

    inner_product = mapped_cord.as_matrix() @ theta

    decision = mapped_cord[np.abs(inner_product) < threshhold]

    return decision.f10, decision.f01
    
data=pd.read_csv('ex2data2.txt',header=None,names=['test1','test2','admit'])
#positive = data[data['admit'].isin([1])]
#negative = data[data['admit'].isin([0])]
#fig, ax = plt.subplots(figsize=(12,8))
#ax.scatter(positive['test1'], positive['test2'], s=50, c='b', marker='o', label='admit')
#ax.scatter(negative['test1'], negative['test2'], s=50, c='r', marker='x', label='unadmit')
#ax.legend()
#ax.set_xlabel('Test 1 Score')
#ax.set_ylabel('Test 2 Score')
#plt.show()

x1=data['test1']
x2=data['test2']
data.insert(3,'Ones',1)
degree=6
for i in range(1,degree):
    for j in range(0,i):
        data['F'+str(i)+str(j)]=np.power(x1,i-j)*np.power(x2,j)
data.drop('test1', axis=1, inplace=True)
data.drop('test2', axis=1, inplace=True)
y=data.iloc[:,0:1]
x=data.iloc[:,1:data.shape[1]]
#theta=np.ones(3)*0.01
theta=np.zeros(x.shape[1])
lamda=0.1
J=cost(theta,x,y,lamda)
dc=gradDec(theta,x,y,lamda)

result2 = opt.fmin_tnc(func=cost, x0=theta, fprime=gradDec, args=(x, y, lamda))
y2 = np.array(y.values)

theta_min = np.matrix(result2[0])
predictions = predict(theta_min, x)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y2)]
accuracy = (sum(map(int, correct)) % len(correct))
print ('accuracy = {0}%'.format(accuracy))

density = 1000
threshhold = 2 * 10**-3

final_theta = theta_min#feature_mapped_logistic_regression(power, l)
x, y = find_decision_boundary(density, degree, final_theta, threshhold)

df = pd.read_csv('ex2data2.txt', names=['test1', 'test2', 'accepted'])
sns.lmplot('test1', 'test2', hue='accepted', data=df, size=6, fit_reg=False, scatter_kws={"s": 100})

plt.scatter(x, y, c='R', s=10)
plt.title('Decision boundary')
plt.show()


