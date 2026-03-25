import tensorflow as tf
from tensorflow.keras import datasets, layers, models, preprocessing

#     ПІДГОТОВКА ДАНИХ
max_words = 10000
# Завантажуємо дані IMDb (лише 10 000 найпопулярніших слів)
(x_train_full, y_train_full), (x_test_full, y_test_full) = datasets.imdb.load_data(num_words=max_words)

def get_data(maxlen=200):
    """Функція для приведення послідовностей до однакової довжини"""
    x_train = preprocessing.sequence.pad_sequences(x_train_full, maxlen=maxlen)
    x_test = preprocessing.sequence.pad_sequences(x_test_full, maxlen=maxlen)
    return x_train, x_test

def build_and_train(layer_type="LSTM", embed_dim=64, maxlen=200, epochs=3):
    """Універсальна функція для створення та навчання моделі"""
    x_train, x_test = get_data(maxlen)
    
    model = models.Sequential([
        # Шар вкладення (Embedding): перетворює індекси слів у вектори
        layers.Embedding(input_dim=max_words, output_dim=embed_dim, input_length=maxlen),
    ])
    
    # Вибір типу рекурентного шару
    if layer_type == "LSTM":
        model.add(layers.LSTM(64))
    elif layer_type == "GRU":
        model.add(layers.GRU(64))
        
    # Вихідний шар для бінарної класифікації (позитивний/негативний відгук)
    model.add(layers.Dense(1, activation="sigmoid"))
    
    model.compile(optimizer="adam", 
                  loss="binary_crossentropy", 
                  metrics=["accuracy"])
    
    print(f"\n--- Навчання: {layer_type}, Embed Dim: {embed_dim}, MaxLen: {maxlen} ---")
    
    # Навчаємо модель 3 епохи
    model.fit(x_train, y_train_full, epochs=epochs, batch_size=64, 
              validation_split=0.2, verbose=1)
    
    # Оцінюємо точність на тестових даних
    _, accuracy = model.evaluate(x_test, y_test_full, verbose=0)
    return accuracy

# ЗАВДАННЯ 1:
# Використовуємо однакові параметри для обох типів шарів
acc_lstm = build_and_train(layer_type="LSTM", embed_dim=64, maxlen=200)
acc_gru = build_and_train(layer_type="GRU", embed_dim=64, maxlen=200)

# ЗАВДАННЯ 2:
# Порівнюємо точність при збільшенні розмірності вектора з 64 до 128
acc_embed_128 = build_and_train(layer_type="LSTM", embed_dim=128, maxlen=200)

# ЗАВДАННЯ 3:
# Перевіряємо, як вплине скорочення тексту відгуку з 200 до 100 слів
acc_len_100 = build_and_train(layer_type="LSTM", embed_dim=64, maxlen=100)


print("\n" + "="*40)
print("АНАЛІЗ РЕЗУЛЬТАТІВ:")
print(f"1. LSTM vs GRU:        LSTM = {acc_lstm:.4f}, GRU = {acc_gru:.4f}")
print(f"2. Embed (64 vs 128):  64 = {acc_lstm:.4f}, 128 = {acc_embed_128:.4f}")
print(f"3. MaxLen (200 vs 100): 200 = {acc_lstm:.4f}, 100 = {acc_len_100:.4f}")
print("="*40)