import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets

#Завдання 3
# Завантажуємо CIFAR-10
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Нормалізація: ділимо на 255.0, щоб значення були від 0 до 1
train_images, test_images = train_images / 255.0, test_images / 255.0

model = tf.keras.Sequential([
    # Вхідний шар: тепер 32x32 і 3 канали (RGB)
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'), 
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax') # 10 класів об'єктів
])

# Завантаження набору Fashion-MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
# Нормалізація зображень
x_train = x_train / 255.0
x_test = x_test / 255.0
# Додавання каналу (1) для згортки
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

#Побудова моделі CNN
# model = tf.keras.Sequential([
# tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(28, 28, 1)),

# tf.keras.layers.MaxPooling2D((2,2)),
# tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
# tf.keras.layers.MaxPooling2D((2,2)),
# tf.keras.layers.Flatten(),
# tf.keras.layers.Dense(64, activation="relu"),
# tf.keras.layers.Dense(10, activation="softmax")
# ])

#Завдання 1 (В результаті підвищилася точність моделі та збільшується час навчання, Узагальнення може погіршуватися)
# model = tf.keras.Sequential([
#     # Первый блок
#     tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
#     tf.keras.layers.MaxPooling2D((2,2)),

#     # Второй блок
#     tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
#     tf.keras.layers.MaxPooling2D((2,2)),

#     # Твой НОВЫЙ блок (добавлен здесь)
#     tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'), 
#     tf.keras.layers.MaxPooling2D((2,2)),

#     # Переход к полносвязной сети
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(10, activation='softmax')
# ])

#Завдання 2 (Це збільшить деталізацію до 256, час навчання збільшується, але це збільшить час навчання моделі)
# model = tf.keras.Sequential([
#     # Первый блок: увеличили с 32 до 64
#     tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
#     tf.keras.layers.MaxPooling2D((2,2)),

#     # Второй блок: увеличили с 64 до 128
#     tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
#     tf.keras.layers.MaxPooling2D((2,2)),

#     # Третий блок: увеличили до 256
#     tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'), 
#     tf.keras.layers.MaxPooling2D((2,2)),

#     # Переход к полносвязной сети
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(128, activation='relu'), # Также увеличил плотный слой для баланса
#     tf.keras.layers.Dense(10, activation='softmax')
# ])

model.compile(optimizer="adam",
loss="sparse_categorical_crossentropy",
metrics=["accuracy"])
model.summary()

#4. Навчання моделі та оцінка якості
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

#5. Візуалізація фільтрів та результатів
# Графік точності
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()