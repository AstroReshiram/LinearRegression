import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_data(x,y):
    plt.figure()
    plt.plot(x, y, 'rx', MarkerSize=10);
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')

def compute_cost(X, y, theta):
    m = len(y)
    h = (theta.T @ X.T).T
    J = 1 / (2 * m) * sum((h - y)**2)
    return J

def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros((num_iters, 1))
    for iter in range(1,num_iters):
        h =  (theta.T @ X.T).T
        dJ = X.T @ (h-y)
        theta = theta - alpha * 1/m * dJ

    J_history[iter] = compute_cost(X, y, theta)
    return theta, J_history

def main():
    data = pd.read_csv('ex1data1.txt', sep=',', header=None).to_numpy()
    X = data[:, 0].reshape((-1,1))
    y = data[:, 1].reshape((-1,1))
    plot_data(X,y)


    m = len(X)
    X = np.hstack((np.ones((m, 1)), X))
    theta = np.zeros((2, 1))
    iterations = 1500
    alpha = 0.01
    J = compute_cost(X, y, theta)
    [theta, J_history] = gradient_descent(X, y, theta, alpha, iterations)
    print(theta)

    plt.plot(X[:,1], X @ theta, '-')
    plt.legend(['Training data', 'Linear regression'])
    plt.show()

if __name__ == "__main__":
    main()