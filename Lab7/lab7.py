import tensorflow as tf
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
import optuna

# Завантаження даних
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#3. Grid Search і Randomized Search
param_grid = {
"mlp__hidden_layer_sizes": [(50,), (100,)],
"mlp__alpha": [0.0001, 0.001, 0.01],
"mlp__learning_rate_init": [0.001, 0.01],
"mlp__batch_size": [16, 32, 64]
}

pipe = Pipeline([
("scaler", StandardScaler()),
("mlp", MLPClassifier(max_iter=500, random_state=42))
])

gs = GridSearchCV(pipe, param_grid, cv=3)
gs.fit(X_train, y_train)
print("Найкращі параметри (GridSearch):", gs.best_params_)

#4. Оптимізація з Optuna
def objective(trial):
    alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
    lr = trial.suggest_float("learning_rate_init", 1e-4, 1e-1, log=True)
    hidden = trial.suggest_categorical("hidden_layer_sizes", [(32,), (64,), (128,)])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    clf = Pipeline([
    ("caler", StandardScaler()),
    ("lp", MLPClassifier(hidden_layer_sizes=hidden,
    alpha=alpha,
    learning_rate_init=lr,
    batch_size=batch_size,
    max_iter=500,
    random_state=42))
    ])
    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
print("Найкращі параметри (Optuna):", study.best_params)

