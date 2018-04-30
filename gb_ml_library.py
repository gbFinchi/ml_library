import numpy as np


def gradient_descent(cost_func, x, y, initial_theta, num_iterations, alpha):
    theta = initial_theta
    for iteration in range(num_iterations):
        cost, grad = cost_func(x, y, theta)
        theta = theta - alpha * grad
    return theta


def normal_equations(x, y):
    return (np.linalg.inv(x.T.dot(x)).dot(x.T)).dot(y)


def feature_normalization(x):
    m = x.shape[1]
    for i in range(m):
        std = x[:, i].std()
        mean = x[:, i].mean()
        x[:, i] = (x[:, i] - mean)/std
    return x


class LinearRegression:
    def __init__(self):
        self.theta = []

    @staticmethod
    def cost_function(x, y, theta):
        hypothesis = x.dot(theta) - y
        m = len(y)
        cost = (hypothesis ** 2).sum() / (2 * m)
        grad = np.zeros(shape=theta.shape)
        for i in range(len(theta)):
            grad[i] = (hypothesis * x[:, i]).sum() / m
        return cost, grad

    def fit(self, x, y, num_iterations=100, alpha=0.01, feature_norm=False, normal_eq=False):
        if feature_norm:
            x = feature_normalization(x)
        ones = np.ones(shape=(x.shape[0],))
        revisited_x = np.column_stack((ones, x))
        if normal_eq:
            self.theta = normal_equations(revisited_x, y)
        else:
            self.theta = np.zeros(shape=(revisited_x.shape[1],))
            self.theta = gradient_descent(self.cost_function, revisited_x, y, self.theta, num_iterations, alpha)

    def predict(self, x):
        ones = np.ones(shape=(x.shape[0],))
        revisited_x = np.column_stack((ones, x))
        return revisited_x.dot(self.theta)
