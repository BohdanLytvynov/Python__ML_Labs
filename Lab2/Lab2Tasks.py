import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def sgd_linear_regression(X, y, learning_rate=0.01, epochs=100):
    # random initial params
    w = 0.0
    b = 0.0
    n = len(X)
    
    # Weight list
    history = []

    for epoch in range(epochs):
        # Permutate data during each epoch
        indices = np.random.permutation(n)
        X = X[indices]
        y = y[indices]
        
        for i in range(n):
            # 1. Take 1 example
            xi = X[i]
            yi = y[i]
            
            # 2. Prediction for each example
            y_pred = w * xi + b
            
            # 3. Calculate error
            error = y_pred - yi
            
            grad_w = 2 * error * xi
            grad_b = 2 * error
            
            # 4. Update Parametrs
            w = w - learning_rate * grad_w
            b = b - learning_rate * grad_b
            
            # Save to history
            history.append((w, b))
            
    return w, b, np.array(history)

def plot_loss_3d(X, y):
    """
    Builds 3D Plot of the loss function 
    X: Input Data
    y: Values (Actual results of the Function)
    """    
    w_range = np.linspace(-10, 10, 100)
    b_range = np.linspace(-10, 10, 100)
    W, B = np.meshgrid(w_range, b_range)
    
    Z = np.zeros(W.shape)
    
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            w_curr = W[i, j]
            b_curr = B[i, j]
            y_pred = w_curr * X + b_curr
            Z[i, j] = np.mean((y_pred - y)**2)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(W, B, Z, cmap='viridis', edgecolor='none', alpha=0.8)
     
    ax.set_xlabel('Weight (w)')
    ax.set_ylabel('Biass (b)')
    ax.set_zlabel('Loss (MSE)')
    ax.set_title('Surface of the loss function')
    
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

print("Task 1 - Calculate gradient of the f(x,y) = x^2 + 3y^2 - 2xy + 4x - 5y")

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

xl = np.linspace(-10, 10, 100)
yl = np.linspace(-10, 10, 100)
x, y = np.meshgrid(xl, yl)

z = x**2 + 3*y**2 - 2*x*y + 4*x - 5*y

#Gradient I just do it on paper: Grad(z) = [ dz/dx, dz/dy ]

dx = 2*x - 2*y + 4
dy = 6*y - 2*x - 5
Z = np.zeros_like(z)

surf = ax.plot_surface(x,y,z,cmap='viridis', edgecolor='none', alpha=0.3)

skip = (slice(None, None, 10), slice(None, None, 10))

ax.quiver(x[skip],y[skip],z[skip],dx[skip],dy[skip],Z[skip], length=0.2, color='red', pivot='tail', arrow_length_ratio=0.8, linewidth=1.5)

fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

ax.set_title("3D Surface and Gradient")
plt.show()

print("Task 2 - Draw function for 3D plot for the Linear Regression")

X_example = np.array([1, 2, 3, 4, 5]) # y = 2*x
y_example = 2 * X_example + np.random.randn(5) # random noise

plot_loss_3d(X_example, y_example)

print("Task 3 - SGD for linear regretion ")

X_data = np.array([1, 2, 3, 4, 5])
y_data = np.array([2.1, 3.9, 6.2, 8.1, 10.3]) # y = 2x

w_final, b_final, hist = sgd_linear_regression(X_data, y_data)
print(f"Results: w = {w_final:.2f}, b = {b_final:.2f}")