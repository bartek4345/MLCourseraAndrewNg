import numpy as np
import matplotlib.pyplot as plt    
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import cm
import itertools


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

#Making 3dplot
axX = np.linspace(-10, 10, 50)
axY = np.linspace(-1, 4, 50)
xx, yy = np.meshgrid(axX, axY, indexing='xy')
zz = np.zeros((axX.size, axY.size))


for (i,j),v in np.ndenumerate(zz):
    zz[i,j] = cost_fun(startTheta= [[xx[i, j]], [yy[i, j]]], X = X, Y = Y)


fig = plt.figure(figsize=(15,6))
ax1 = fig.add_subplot(122)
ax2 = fig.add_subplot(121, projection='3d')

# Left plot
ax1.contour(xx, yy, zz, np.logspace(-2, 3, 20), cmap=plt.cm.jet)
ax1.scatter(theta[0],theta[1], c='r')

#Right plot
ax2.plot_surface(xx, yy, zz, rstride = 1, cstride = 1, alpha = 0.6, cmap = plt.cm.jet)
ax2.view_init(elev=20, azim=220)
ax2.set_zlabel('Cost', fontsize = 25)

for ax in fig.axes:
    ax.set_xlabel(r'$\theta_0$', fontsize = 25)
    ax.set_ylabel(r'$\theta_1$', fontsize = 25)
    

#Linear regression with multiple variables

data = np.loadtxt('ex1data2.txt', delimiter=",")
data.shape
X = np.c_[np.ones(data.shape[0]), data[:,0:-1]]
Y = np.c_[data[:,-1]]

#Feature Normalization

X_fn = X.copy()
means, stds = [], []
for col in range(X_fn.shape[1]):
    means.append(np.mean(X[:, col]))
    stds.append(np.std(X[:, col]))
    if col != 0:
        X_fn[:, col] = (X_fn[:, col] - means[col])/stds[col]

#Gradient Descent

#Ploting the gradient descent steps for different alpha
plt.figure()
for alpha in np.linspace(0.01, 1, 50):
    theta , Cost_J = gradient_descent(theta = np.zeros([X_fn.shape[1], 1]), X = X_fn, Y = Y, iterations= 50, alpha= alpha)
    plt.plot(Cost_J)

#For alpha 0.05
theta , Cost_J = gradient_descent(theta = np.zeros([X_fn.shape[1], 1]), X = X_fn, Y = Y, iterations= 50, alpha= 0.1)
plt.figure()
plt.plot(Cost_J)

#Predicting price of house with 1650 square feet and 3 bedrooms
ypred = np.array([1650, 3])
ypredscaled = []
for x in range(len(ypred)):
    ypredscaled.append((ypred[x] - means[x+1])/stds[x+1])
ypredscaled.insert(0, 1)
#Result
hip_fun(ypredscaled, theta)

#Normal Equation
def normalEqn(X, y):
    return (np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y))
    
#Result
hip_fun([1, 1650, 3], normalEqn(X,Y))