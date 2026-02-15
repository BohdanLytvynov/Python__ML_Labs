import numpy as np
import matplotlib.pyplot as plt
import keyboard
import time
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def plot_regression_results(X_test, y_test, y_pred, feature_idx=0, title="Порівняння результатів"):
    """
    Візуалізує реальні та прогнозовані значення.
    
    :param X_test: Матриця ознак (NumPy array або Pandas DataFrame)
    :param y_test: Реальні значення цільової змінної
    :param y_pred: Прогнозовані значення моделі
    :param feature_idx: Індекс ознаки для осі X (за замовчуванням 0)
    :param title: Заголовок графіка
    """
    plt.figure(figsize=(10, 6))
    
    # Малюємо реальні значення
    plt.scatter(X_test[:, feature_idx], y_test, color='blue', label='Реальні значення', alpha=0.6)
    
    # Малюємо прогнозовані значення
    plt.scatter(X_test[:, feature_idx], y_pred, color='red', label='Прогнозовані значення', marker='x')
    
    plt.xlabel('Довжина пелюстки (або інша ознака)')
    plt.ylabel('Клас / Значення')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

print("Завантаження данних")
# Завантажуємо набір даних Iris
iris = datasets.load_iris()
X = iris.data # Ознаки (довжина та ширина пелюсток і чашолистків)
y = iris.target # Мітки (0, 1, 2 - три різні види квітів)
print("Данні завантажені.")
# Розбиваємо дані на тренувальні (80%) і тестові (20%) набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Данні готові.")

while True:
    print("Оберіть тип регресії: \n \t-1 - Лінійна\n\t-2 - Логістична регресія\n\t-3 - К Найближчі сусіди\n")
    # Чекаємо натискання клавіші вибору (тільки подія KEY_DOWN)
    event = keyboard.read_event()
    if event.event_type == keyboard.KEY_UP:
        continue  # Ігноруємо відпускання клавіш

    com = event.name
    if com == "1":
        # Створюємо та навчаємо модель Лінійна
        linear_model = LinearRegression()
        linear_model.fit(X_train[:, 0].reshape(-1, 1), y_train)    
        # Передбачення
        y_pred_lr = linear_model.predict(X_test[:, 0].reshape(-1, 1))
        # Візуалізація
        plot_regression_results(X_test, y_test, y_pred_lr, feature_idx=0, title="Лінійна регресія: Iris")
        print("Графік намальовано")        
    elif com == "2":
        # Створюємо та навчаємо модель Логістична
        logistic_model = LogisticRegression(max_iter=200)
        logistic_model.fit(X_train, y_train)
        # Передбачення
        y_pred_logistic = logistic_model.predict(X_test)
        accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
        print(f'Точність логістичної регресії: {accuracy_logistic:.2f}')
        # Візуалізація
        plot_regression_results(X_test, y_test, y_pred_logistic, feature_idx=0, title="Логістична регресія: Iris")
        print("Графік намальовано")
    elif com == "3":
        # Створюємо та навчаємо модель К Сусіди
        knn_model = KNeighborsClassifier(n_neighbors=5)
        knn_model.fit(X_train, y_train)
        y_pred_knn = knn_model.predict(X_test)
        accuracy_knn = accuracy_score(y_test, y_pred_knn)
        print(f'Точність KNN: {accuracy_knn:.2f}')
        plot_regression_results(X_test, y_test, y_pred_knn, feature_idx=0, title="К сусіди регресія: Iris")
        print("Графік намальовано")
    else:
        print("Невірна команда!")    
    
    print("Якщо ви бажаєте вийти натисніть ESC, а якщо бажаєте продовжити роботи тисніть будь яку клавішу.")

    while True:
        next_step = keyboard.read_event()
        if next_step.event_type == keyboard.KEY_DOWN:
            if next_step.name == 'esc':
                print("Завершення роботи...")
                exit() # Повний вихід
            else:
                # Будь-яка інша клавіша — виходимо з внутрішнього циклу до меню
                print("Повернення до меню...")
                time.sleep(0.2) # Пауза, щоб не "проскочити" вибір у меню
                break
    



