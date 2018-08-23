import numpy as np
import matplotlib.pyplot as plt    
import pandas as pd
from scipy.special import expit
from scipy import optimize

#loading dataset
data = np.loadtxt('ex2data1.txt', delimiter= ',')
y = np.array(data[:, -1])
X = np.array(data[:, 0:-1])
X = np.insert(X, 0, 1, axis = 1)
m = y.size
#Making scatterlot
passedData = data[:, 2] == 1
failedData = data[:, 2] == 0

axes = plt.gca()
axes.scatter(data[passedData][:, 0], data[passedData][:, 1], c = 'black', label = "Admitted")
axes.scatter(data[failedData][:, 0], data[failedData][:, 1], c = 'yellow', label = "Not admitted")
axes.set_xlabel('Exam 1 score')
axes.set_ylabel('Exam 2 score')
axes.legend(loc = 3)

#Implementation of hypothesis function
def hypFun(theta, X):
    return expit(np.dot(X,theta))    

#Implementation of Cost function
def costFun(X, y, theta):
    regPos = np.dot(-y, np.log(hypFun(theta,X)))
    regNeg = np.dot(1-y,np.log(1 - hypFun(theta,X)))
    Cost = float((1./y.shape[0]) * (np.sum(regPos - regNeg)))
    #grad = 1 / y.shape[0] * (hypFun(theta,X) - y).dot(X)
    return Cost
#Testing cost function - should be about 0.69
costFun(X, y, np.zeros((X.shape[1], 1)))

result = optimize.fmin(costFun, x0=np.zeros((X.shape[1],1)), args=(X, y), maxiter=400, full_output=True)
result

