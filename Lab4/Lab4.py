import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Генерація синтетичних даних для регресії
X = np.linspace(-1, 1, 100)
y = 3 * X + np.random.randn(*X.shape) * 0.3

# Побудова моделі в TensorFlow
model = tf.keras.Sequential([
tf.keras.layers.Dense(1, input_shape=(1,))
])

# Реалізація навчання з використанням різних оптимізаторів
def train_model(optimizer, name):
    model.compile(optimizer=optimizer, loss="mae")# 1 Міняємо mse на mae
    history = model.fit(X, y, epochs=100, verbose=0)
    plt.plot(history.history["loss"], label=name)

# Кастомна реалізація Градієнтного Спуску
def linear_regression_gd(X, y, learning_rate=0.01, epochs=100):
    # Ініціалізація ваг
    w = 0.0
    b = 0.0
    n = float(len(X))
    
    # Словник для збереження історії навчання
    history = {'loss': [], 'w': [], 'b': []}
    
    for i in range(epochs):
        # Прогноз
        y_pred = w * X + b
        
        # Розрахунок помилки (MSE)
        loss = np.mean((y - y_pred)**2)
        
        # Розрахунок градієнтів
        dw = (-2 / n) * np.sum(X * (y - y_pred))
        db = (-2 / n) * np.sum(y - y_pred)
        
        # Оновлення параметрів
        w -= learning_rate * dw
        b -= learning_rate * db
        
        # Запис в історію
        history['loss'].append(loss)
        history['w'].append(w)
        history['b'].append(b)
        
    return w, b, history


    # Реалізація навчання з використанням різних оптимізаторів
def train_model_custom(optimizer, name):
    model.compile(optimizer=optimizer, loss="mae")# 1 Міняємо mse на mae
    history = linear_regression_gd(X, y, learning_rate= 0.1, epochs=100)
    plt.plot(history[2]["loss"], label=name)

# # SGD
# train_model(tf.keras.optimizers.SGD(learning_rate=0.1), "SGD")
# # SGD + momentum
# train_model(tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9), "SGD + Momentum")
# # Adam
# train_model(tf.keras.optimizers.Adam(learning_rate=0.1), "Adam")
# # RMSprop
# train_model(tf.keras.optimizers.RMSprop(learning_rate=0.1), "RMSprop")
# # Adagrad
# train_model(tf.keras.optimizers.Adagrad(learning_rate=0.1), "Adagrad")
# Кастомний тренінг моделей 
train_model_custom(tf.keras.optimizers.Adagrad(learning_rate=0.1), "Adagrad")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.title("Збіжність оптимізаторів")
plt.show()