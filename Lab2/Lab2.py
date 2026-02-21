import numpy as np
import matplotlib.pyplot as plt

def num_der(func, x, h=1e-5):
    return (func(x + h) - func(x - h))/(2 * h)

def fun1(x): return x**2 + 3*x + 5

# vector = np.array([1,2,3])
# mat = np.array([
#     [1,2],
#     [3,4]
# ])

# print("Vector: ", vector)
# print("Mat: ", mat)
# print("Transposed Mat:\n", mat.T)
# print("Mat * Vect:", mat @ vector[0:2:1])
# print("Mat * Vect:", mat.dot(vector[0:2:1]))

# x = np.linspace(-10, 10, 100)

# y = x*x
# dy = 2*x

# plt.plot(x, y, label="y = x^2")
# plt.plot(x, dy, label="y = 2*x", linestyle="--")
# plt.legend()
# plt.scatter(0, 0, color="red", s=20)
# plt.xlabel("x")
# plt.ylabel("y")
# plt.grid(True)
# plt.show()

# grad = num_der(fun1, x)
# print("Gradients: ",grad)

# plt.plot(x, fun1(x), label="Function")
# plt.plot(x, grad, label="Gradient", linestyle="--")
# plt.legend()
# plt.grid(True)
# plt.show()

X = np.array([1,2,3,4,5]) #y = 2*x
Y = np.array([2,4,6,8,10])
w = 0 #yp = w*X + b - linear regretion
b = 0
l_r = 0.01
epochs = 1000

for epoch in range(epochs):
    yp = w*X + b
    loss = np.mean((Y - yp)**2)
    dw = -2*np.mean(X * (Y - yp))
    db = -2*np.mean(Y - yp)

    w -= dw*l_r
    b -= db*l_r

    if epoch % 10 == 0: 
        print(f"Epoch: {epoch}: Loss = {loss:.4f}")
    
print(f"w = {w:.2f}, b = {b:.2f}")

# Візуалізація
plt.scatter(X, Y, label="Data")
plt.plot(X, w * X + b, color="red", label="Line")
plt.legend()
plt.show()

