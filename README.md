# ml assignment
# student info: Sujan Akena, 700770399, CRN: 11595



# This code shows how to do linear regression in two ways — closed-form and gradient descent.

#First, it makes some fake data using y = 3 + 4x + noise.

#Then it uses the normal equation to find the best-fit line (gives slope and intercept directly).

#After that, it runs gradient descent starting from [0,0] and updates the values until they match the closed-form result.

#Finally, it plots the data, both lines, and the loss curve to show that gradient descent actually works.

# program
import numpy as np
import matplotlib.pyplot as plt

# Make random data: y = 3 + 4x + some noise
rng = np.random.default_rng(42)
n = 200
x = rng.uniform(0.0, 5.0, size=n)
epsilon = rng.normal(0.0, 1.0, size=n)
y = 3.0 + 4.0 * x + epsilon
X = np.column_stack([np.ones_like(x), x])  # add column of 1s for bias

# Solve using normal equation (closed-form)
theta_closed = np.linalg.inv(X.T @ X) @ X.T @ y
b0_closed, b1_closed = theta_closed

# Gradient descent setup
def mse(theta, X, y):
    return np.mean((y - X @ theta) ** 2)

def grad_mse(theta, X, y):
    n = X.shape[0]
    return (-2.0 / n) * (X.T @ (y - X @ theta))

theta_gd = np.array([0.0, 0.0])
eta, num_iters = 0.05, 1000
loss_history = []
for _ in range(num_iters):
    loss_history.append(mse(theta_gd, X, y))
    theta_gd -= eta * grad_mse(theta_gd, X, y)
b0_gd, b1_gd = theta_gd

# Print both results to compare
print("Closed-form solution:")
print(f"  Intercept (b0): {b0_closed:.6f}")
print(f"  Slope     (b1): {b1_closed:.6f}")
print("\nGradient Descent solution:")
print(f"  Intercept (b0): {b0_gd:.6f}")
print(f"  Slope     (b1): {b1_gd:.6f}")

# Plot data + both lines
x_line = np.linspace(x.min(), x.max(), 200)
X_line = np.column_stack([np.ones_like(x_line), x_line])

plt.figure()
plt.scatter(x, y, s=12, alpha=0.7, label="Data")
plt.plot(x_line, X_line @ theta_closed, label="Closed-form fit", linewidth=2)
plt.plot(x_line, X_line @ theta_gd, label="GD fit", linestyle="--", linewidth=2)
plt.xlabel("x"); plt.ylabel("y")
plt.title("Linear Regression: Closed-form vs Gradient Descent")
plt.legend(); plt.show()

# Plot loss curve
plt.figure()
plt.plot(loss_history)
plt.xlabel("Iteration"); plt.ylabel("MSE")
plt.title("Gradient Descent Loss Curve")
plt.show()

