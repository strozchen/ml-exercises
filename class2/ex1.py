# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 14:39:05 2018
Logistic Regression
@author: admin
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import seaborn as sns
#sigmoid函数
def sigmoid(z):
    return 1/(1+np.exp(-z))
#画出sigmoid函数曲线
def plotsigmoid():
    nums=np.arange(-10,10,step=1)
    fig,ax=plt.subplots()
    ax.plot(nums,sigmoid(nums),'r')
    plt.show()
#代价函数
def cost(theta,X,y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first=np.multiply(-y,np.log(sigmoid(X*theta.T)))
    second=np.multiply((1-y),np.log(1-sigmoid(X*theta.T)))
    return np.sum(first-second)/(len(X))
#梯度下降,此处只计算一个周期，后续使用SciPy's truncated newton寻优
def gradientDec(theta,X,y):
    theta=np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    
    temp=int(theta.ravel().shape[1])
    grad=np.zeros(temp)
    error=sigmoid(X*theta.T)-y
    for i in range(temp):
        term=np.multiply(error,X[:,i])
        grad[i]=np.sum(term)/len(X)
    return grad
#计算预测结果
def predict(theta,X):
    predictrs=sigmoid(X*theta.T)
    return [1 if x>=0.5 else 0 for x in predictrs]
#读数据
data=pd.read_csv('ex2data1.txt',header=None,names=['score1','score2','passed'])
#print(data.head())
#分离数据
cols=data.shape[1]
score1=data.iloc[:,0:1]
score2=data.iloc[:,1:2]
passed=data.iloc[:,cols-1:cols]
#转为矩阵
score1=np.matrix(score1.values)
score2=np.matrix(score2.values)
passed=np.matrix(passed.values)

#plotsigmoid()

theta=np.zeros(3)
X=np.column_stack((np.ones(score1.shape),score1,score2))
#rs=cost(theta,X,passed)
#rs=gradientDec(theta,X,passed)
#用SciPy's truncated newton（TNC）实现寻找最优参数。
rs=opt.fmin_tnc(func=cost,x0=theta,fprime=gradientDec,args=(X,passed))
print(rs)
print(cost(rs[0],X,passed))

theta_min=np.matrix(rs[0])
predictions=predict(theta_min,X)
correct=0
uncorrect=0


#d1保存预测正确的点，d2保存预测错误的点
#co_ex1=[0]
#co_ex2=[0]
#unco_ex1=[0]
#unco_ex2=[0]
#r=X[:,1:]
#r=np.array(r)
#for (a,b,c) in zip(predictions,passed,r):
for (a,b) in zip(predictions,passed):
    if (a==1 and b[0,0]==1) or (a==0 and b[0,0]==0):
        correct+=1
#        co_ex1.append(c[0])
#        co_ex2.append(c[1])
    else:
        uncorrect+=1
#        unco_ex1.append(c[0])
#        unco_ex2.append(c[1])
#accuracy=correct/(correct+uncorrect)
#print(accuracy)
#co_ex1.remove(0)
#co_ex2.remove(0)
#d1 = {'score1': co_ex1, 'score2':co_ex2}
#co1=pd.DataFrame(data=d1)
#unco_ex1.remove(0)
#unco_ex2.remove(0)
#d2={'score1':unco_ex1,'score2':unco_ex2}
#co2=pd.DataFrame(data=d2)

#找到分隔线
coef = -(rs[0] / theta_min[0,2])  # find the equation
print(coef)

x = np.arange(130, step=0.1)
y = coef[0] + coef[1]*x

#显示读到的数据
positive=data[data['passed'].isin([1])]
negative=data[data['passed'].isin([0])]
fig,ax=plt.subplots(figsize=(12,8))
ax.scatter(positive['score1'],positive['score2'],s=50,c='b',marker='o',label='passed')
ax.scatter(negative['score1'],negative['score2'],s=50,c='r',marker='x',label='unpassed')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
plt.show()


sns.set(context="notebook", style="ticks", font_scale=1.5)
sns.lmplot('score1', 'score2', hue='passed', data=data, 
           size=6, 
           fit_reg=False, 
           scatter_kws={"s": 25}
          )

plt.plot(x, y, 'grey')
plt.xlim(0, 130)
plt.ylim(0, 130)
plt.title('Decision Boundary')
plt.show()




