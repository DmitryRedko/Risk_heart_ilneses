import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
import pickle
from database import save_model_in_db
"""
pandas (pd) - библиотека для анализа и обработки данных в формате таблицы;
numpy (np) - библиотека для работы с массивами и матрицами;
train_test_split из sklearn.model_selection - функция для разделения данных на обучающую и тестовую выборки;
StandardScaler из sklearn.preprocessing - класс для масштабирования признаков;
GridSearchCV из sklearn.model_selection - класс для перебора параметров модели и выбора наилучших;
LogisticRegression из sklearn.linear_model - класс для логистической регрессии;
KNeighborsClassifier из sklearn.neighbors - класс для k-ближайших соседей;
SVC из sklearn.svm - класс для метода опорных векторов;
GaussianNB из sklearn.naive_bayes - класс для наивного байесовского классификатора;
DecisionTreeClassifier из sklearn.tree - класс для дерева решений;
RandomForestClassifier и RandomForestRegressor из sklearn.ensemble - классы для случайного леса;
accuracy_score из sklearn.metrics - функция для расчета точности (accuracy) модели;
pickle - модуль для сериализации и десериализации объектов Python;
save_model_in_db из database - функция для сохранения модели в базу данных.
"""


class Models:
    def LR(self, X_train, y_train):
        print("Start educating LR model")
        self.name = 'LogisticRegressionModel'
        lr = LogisticRegression(max_iter=100000)
        param_grid = {
            'C': [0.0001, 0.001, 0.01, 0.1, 1.0, 2.0, 3.0, 10, 50, 100, 1000, 10000]
        }
        grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        best = grid_search.best_estimator_
        print("LR model is educated")
        return {'model': best, 'name': self.name}

    def KNN(self, X_train, y_train):
        print("Start educating KNN model")
        self.name = 'KNNeighborsModel'
        knn = KNeighborsClassifier()
        param_grid = {'n_neighbors': [3, 5, 7],
                      'weights': ['uniform', 'distance']}
        grid_search = GridSearchCV(knn, param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        best = grid_search.best_estimator_
        print("KNN model is educated")
        return {'model': best, 'name': self.name}

    def KernelSVM(self, X_train, y_train):
        print("Start educating SVM model")
        self.name = "KernaelSVM"
        # Задаем параметры для GridSearchCV
        param_grid = {'C': [0.1, 1, 10, 15],
                      'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                      'gamma': ['scale', 'auto'] + list(np.logspace(-3, 3, 7))}

        # Создаем объект SVM и объект GridSearchCV для поиска наилучших параметров
        svm = SVC(probability=True)
        grid_search = GridSearchCV(svm, param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        best = grid_search.best_estimator_
        print("SVM model is educated")
        return {'model': best, 'name': self.name}

    def NBayes(self, X_train, y_train):
        print("Start educating NBayes model")
        self.name = "NBayes"
        gnb = GaussianNB()
        gnb.fit(X_train, y_train)
        print("NBayes model is educated")
        return {'model': gnb, 'name': self.name}

    def tree(self, X_train, y_train):
        print("Start educating Tree model")
        self.name = "Tree"
        param_grid = {
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        }
        dt_classifier = DecisionTreeClassifier()

        # Perform grid search cross-validation to find the best parameters
        grid_search = GridSearchCV(dt_classifier, param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        best = grid_search.best_estimator_
        print("Tree model is educated")
        return {'model': best, 'name': self.name}

    def RFC(self, X_train, y_train):
        print("Start educating RFC model")
        self.name = "RandomForestClassifier"
        param_grid = {
            'n_estimators': [5, 10, 15],
            'max_depth': [2, 4, 7],
            'min_samples_split': [2, 5, 7],
            'min_samples_leaf': [1, 2, 3]
        }
        # Create a Random Forest Classifier
        rfc = RandomForestClassifier()
        grid_search = GridSearchCV(rfc, param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        best = grid_search.best_estimator_
        print("RFC model is educated")
        return {'model': best, 'name': self.name}

    def RFR(self, X_train, y_train):
        print("Start educating RFR model")
        self.name = "RandomForestRegressor"
        param_grid = {
            'n_estimators': [5, 10, 15],
            'max_depth': [5, 10],
            'max_features': ['auto', 'sqrt'],
            'min_samples_split': [2, 3, 4],
            'min_samples_leaf': [1, 2, 4]
        }
        rfr = RandomForestRegressor()
        grid_search = GridSearchCV(
            estimator=rfr, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        best = grid_search.best_estimator_
        print("RFC model is educated")
        return {'model': best, 'name': self.name}


"""Класс для реализации моделей машинного обучения. В данном случае методы класса - это модели.
Все методы устроены примерно одинаково. Сначла задается param_grid, в котором перечислены возможные 
параметры модели, затем объявляется сам метод машинного обучения. grid_search подбирает наилучшие 
параметры для модели, лучшая из моделей возвращается из функции"""


"""Функция NormalaizeTrainTest(X_train, X_test) принимает на вход два набора данных - тренировочный 
X_train и тестовый X_test. Данные обрабатываются с помощью StandardScaler() - это метод масштабирования 
признаков, который масштабирует каждый признак таким образом, чтобы среднее значение признака было равно 
0, а стандартное отклонение было равно 1. Признаки, которые подвергаются масштабированию, перечислены в 
списке num_cols. Функция возвращает два набора данных - обработанные X_train и X_test. Также создается
 файл scaler.pkl, в котором сохраняется масштабирующий объект scaler с помощью библиотеки pickle."""


def NormalaizeTrainTest(X_train, X_test):
    scaler = StandardScaler()
    num_cols = ['age', 'creatinine_phosphokinase', 'ejection_fraction',
                'platelets', 'serum_creatinine', 'serum_sodium', 'time']
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    with open('scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)
    return X_train, X_test


"""Функция createTrainTest(df, target) принимает на вход два параметра - набор данных df
 и имя целевого столбца target. Набор данных разделяется на тренировочный и тестовый с помощью
  метода train_test_split(). Тренировочный набор данных возвращается как X_train, а 
  тестовый - как X_test. Возвращаются также целевые значения для тренировочных данных y_train и
   тестовых данных y_test."""


def createTrainTest(df, target):
    X = df.drop(target, axis=1)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

"""Функция predict(data, model) принимает на вход данные data и модель машинного 
обучения model и возвращает прогноз модели для этих данных."""
def predict(data, model):
    y_pred = model.predict(data)
    return y_pred

"""Функция predict_proba(values_list, model) принимает на вход список значений values_list 
и модель машинного обучения model и возвращает вероятность прогнозов модели для этих данных."""
def predict_proba(values_list, model):
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    data = scaler.transform(values_list)
    y_pred = model.predict_proba(data)
    return y_pred

"""Функция accuracy(data, realdata) принимает на вход два параметра - прогнозы data и реальные данные 
realdata и вычисляет точность прогноза с помощью метрики accuracy_score."""
def accuracy(data, realdata):
    accuracy = accuracy_score(realdata, data)
    return accuracy

"""Функция save_model(model, acc, model_name) принимает на вход модель машинного обучения model, точность 
модели acc и имя модели model_name. Модель сохраняется с помощью библиотеки pickle и затем сохраняется 
в базе данных с помощью функции save_model_in_db()."""
def save_model(model, acc, model_name):

    # Сериализуем модель в байты
    model_data = pickle.dumps(model)

    save_model_in_db(model_name, acc, model_data)

"""Функция choose_best(x_test, y_test, models) принимает на вход тестовые данные x_test, реальные значения
 y_test и список моделей models. Функция обучает каждую модель на тренировочных данных и вычисляет точность
  прогноза для тестовых данных. Возвращается модель с наибольшей точностью, сама точность и имя модели."""
def choose_best(x_test, y_test, models):
    best = ''
    m = -1
    name = ''
    for model in models:
        print(model['name'], end=' ')
        pred = predict(x_test, model['model']) > 0.5
        res = accuracy(pred, y_test)
        print(res)
        if (res > m):
            m = res
            best = model['model']
            name = model['name']
    return best, m, name
