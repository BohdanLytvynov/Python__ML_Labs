import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Завантаження даних MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Нормалізація
x_train = x_train / 255.0
x_test = x_test / 255.0

# Перетворення форми для подачі в модель
x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)

# Модель 1 accuracy: 0.9861 - loss: 0.0448 - val_accuracy: 0.9724 - val_loss: 0.0883
# model = models.Sequential([
# layers.Input(shape=(784,)),
# layers.Dense(128, activation="relu"),
# layers.Dense(10, activation="softmax")
# ])

# Модель 2 - Accuracy increased accuracy: 0.9878 - loss: 0.0396 - val_accuracy: 0.9780 - val_loss: 0.0744
# model = models.Sequential([
# layers.Input(shape=(784,)),
# layers.Dense(256, activation="relu"),
# layers.Dense(128, activation="relu"),
# layers.Dense(64, activation="relu"),
# layers.Dense(10, activation="softmax")
# ])

# Модель 3 tanh функция активації Accuracy increased accuracy: 0.9882 - loss: 0.0367 - val_accuracy: 0.9780 - val_loss: 0.0748
# model = models.Sequential([
# layers.Input(shape=(784,)),
# layers.Dense(256, activation="relu"),
# layers.Dense(128, activation="tanh"),
# layers.Dense(64, activation="tanh"),
# layers.Dense(10, activation="softmax")
# ])


# Модель 4 sigmoid Accuracy increased even more accuracy: 0.9893 - loss: 0.0354 - val_accuracy: 0.9789 - val_loss: 0.0703
# model = models.Sequential([
# layers.Input(shape=(784,)),
# layers.Dense(256, activation="relu"),
# layers.Dense(128, activation="sigmoid"),
# layers.Dense(64, activation="sigmoid"),
# layers.Dense(10, activation="softmax")
# ])

# Модель 5 Add Dropout accuracy: 0.9844 - loss: 0.0548 - val_accuracy: 0.9780 - val_loss: 0.0752
# model = models.Sequential([
# layers.Input(shape=(784,)),
# layers.Dense(256, activation="relu"),
# layers.Dense(128, activation="sigmoid"),
# layers.Dropout(0.2),
# layers.Dense(64, activation="sigmoid"),
# layers.Dropout(0.3),
# layers.Dense(10, activation="softmax")
# ])

# Модель 5 Add Dropout accuracy: 
model = models.Sequential([
layers.Input(shape=(784,)),
layers.Dense(256, activation="relu"),
layers.Dense(128, activation="sigmoid"),
layers.Dropout(0.2),
layers.Dense(64, activation="sigmoid"),
layers.Dropout(0.3),
layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam",
loss="sparse_categorical_crossentropy",

metrics=["accuracy"])

model.summary()

history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

#history = model.fit(x_train, y_train, epochs=4, validation_data=(x_test, y_test)) #На 10 % меньше епох - 
#  accuracy: 0.9809 - loss: 0.0695 - val_accuracy: 0.9793 - val_loss: 0.0702

# Побудова графіків
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()