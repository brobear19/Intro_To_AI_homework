import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def gradientDescent(X, Y, theta, learning_rate, iterations):
    m = len(Y)
    cost_history = np.zeros(iterations)
    
    for i in range(iterations):
        predictions = X.dot(theta)
        error = predictions - Y
        gradient = (1/m) * X.T.dot(error)
        theta = theta - learning_rate * gradient
        cost_history[i] = (1/(2*m)) * np.sum((predictions - Y) ** 2)
    return theta, cost_history

def plot(X, Y, theta, feature_name):
    plt.figure(figsize=(8, 5))
    plt.scatter(X, Y, color='blue', label='Data points')
    plt.plot(X, theta[0] + theta[1] * X, color='red', label='Regression line')
    plt.xlabel(feature_name)
    plt.ylabel('y')
    plt.title('Regression model using ' + feature_name)
    plt.legend()
    plt.show()

data = pd.read_csv('D3.csv', header=None)
x1 = data.iloc[:, 0].values
x2 = data.iloc[:, 1].values
x3 = data.iloc[:, 2].values
y  = data.iloc[:, 3].values
m = len(y)


# Problem 1
def univariate_regression(x, y, learning_rate=0.1, iterations=1000):
    m = len(y)
    X = np.vstack([np.ones(m), x]).T
    theta = np.zeros(2)
    theta, cost_history = gradientDescent(X, y, theta, learning_rate, iterations)
    return theta, cost_history

learning_rate_uni = 0.1
iterations = 1000

theta_x1, cost_x1 = univariate_regression(x1, y, learning_rate=learning_rate_uni, iterations=iterations)
theta_x2, cost_x2 = univariate_regression(x2, y, learning_rate=learning_rate_uni, iterations=iterations)
theta_x3, cost_x3 = univariate_regression(x3, y, learning_rate=learning_rate_uni, iterations=iterations)

print("Univariate Models:")
print("For x1: y = {:.4f} + {:.4f} * x1".format(theta_x1[0], theta_x1[1]))
print("For x2: y = {:.4f} + {:.4f} * x2".format(theta_x2[0], theta_x2[1]))
print("For x3: y = {:.4f} + {:.4f} * x3".format(theta_x3[0], theta_x3[1]))

plot(x1, y, theta_x1, 'x1')
plot(x2, y, theta_x2, 'x2')
plot(x3, y, theta_x3, 'x3')

plt.figure(figsize=(8, 5))
plt.plot(range(iterations), cost_x1, label='x1')
plt.plot(range(iterations), cost_x2, label='x2')
plt.plot(range(iterations), cost_x3, label='x3')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost over iterations for univariate regressions')
plt.legend()
plt.show()

print("Final cost values:")
print("x1: {:.4f}".format(cost_x1[-1]))
print("x2: {:.4f}".format(cost_x2[-1]))
print("x3: {:.4f}".format(cost_x3[-1]))

X_multi = np.column_stack((np.ones(m), x1, x2, x3))
theta_multi = np.zeros(4) 

learning_rate_multi = 0.1
theta_multi, cost_history_multi = gradientDescent(X_multi, y, theta_multi, learning_rate_multi, iterations)

print("\nMultivariate Model:")
print("y = {:.4f} + {:.4f} * x1 + {:.4f} * x2 + {:.4f} * x3".format(theta_multi[0], theta_multi[1], theta_multi[2], theta_multi[3]))

plt.figure(figsize=(8, 5))
plt.plot(range(iterations), cost_history_multi, label='Multivariate Cost', color='purple')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost over iterations for multivariate regression')
plt.legend()
plt.show()

# Problem 2:
def predict_multivariate(x_new, theta):
    x_new = np.hstack((np.ones((x_new.shape[0], 1)), x_new))
    return x_new.dot(theta)


new_inputs = np.array([
    [1, 1, 1],
    [2, 0, 4],
    [3, 2, 1]
])
predictions = predict_multivariate(new_inputs, theta_multi)
print("\nPredictions for new input values:")
print("For (1, 1, 1): y = {:.4f}".format(predictions[0]))
print("For (2, 0, 4): y = {:.4f}".format(predictions[1]))
print("For (3, 2, 1): y = {:.4f}".format(predictions[2]))

# -----------------------------------------------------------------------------
# Final Remarks:
# - The code above implements gradient descent without relying on any built-in ML libraries.
# - For each univariate regression, the final model, regression line plot, and cost history plot are produced.
# - The multivariate model is also trained, its cost history plotted, and predictions are made on new data.
# - In your report, be sure to include your name, student ID, assignment number, and a link to your GitHub repository.
# - You should also discuss how changing the learning rate (e.g., using 0.1 vs. 0.01) affected convergence and final cost.
# -----------------------------------------------------------------------------
