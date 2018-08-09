import numpy as np
import matplotlib.pyplot as plt    



data = np.loadtxt('ex1data1.txt', delimiter=',')
data.shape

#Linear regression with one variable
X = np.c_[np.ones(data.shape[0]),data[:,0]]
Y = np.c_[data[:,1]]

#Making scatterplot

plt.figure() #making an empty plot
plt.plot(X[:,1],Y,'o',markersize=10, alpha = 0.5, color = "purple")
plt.grid(True) #Always plot.grid true!
plt.ylabel('Profit in $10,000s')
plt.xlabel('Population of City in 10,000s')

theta = np.zeros([2,1]) #initialize theta

#Hipothesis function
def hip_fun(X, theta):
    return np.dot(X, theta)

#Cost function
def cost_fun(startTheta, X, Y):
    m = Y.size
    h = hip_fun(X, startTheta)
    return 1/(2*m)*np.sum(np.square(h - Y))

#Test of cost function
cost_fun(startTheta = theta, X = X, Y = Y)

#Gradient descent algorithm
def gradient_descent(X, Y, theta = np.zeros([2,1]), alpha = 0.01, iterations = 1500):
    m = Y.size
    CostHist = np.zeros(iterations)
    for iter in np.arange(iterations):
        h = hip_fun(X, theta)
        thetaTemp = theta - alpha*(1/m)*(X.T.dot(h-Y))
        theta = thetaTemp
        CostHist[iter] = cost_fun(theta,X,Y)
    return(theta, CostHist)
    
theta , Cost_J = gradient_descent(X, Y)

#Ploting the gradient descent steps
plt.figure()
plt.plot(Cost_J)

#Plotnig regresion line
plt.figure()
plt.plot(X[:,1], Y[:,0], 'o', alpha = 0.5, markersize=10, label='Training Data')
plt.plot(X[:,1], theta[0] +(theta[1]*X[:,1]),'r-',label = 'Hypothesis: h(x) = %0.2f + %0.2fx'%(theta[0],theta[1]))
plt.grid(True) #Always plot.grid true!
plt.ylabel('Profit in $10,000s')
plt.xlabel('Population of City in 10,000s')
plt.legend()

#Predictions for areas of 35000 and 70000 people
print(theta.T.dot([1, 3.5])*10000)
print(theta.T.dot([1, 7])*10000)



