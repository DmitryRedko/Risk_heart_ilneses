import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import pickle
from database import save_model_in_db


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
            'n_estimators': [5,10,15],
            'max_depth': [2,4,7],
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


def NormalaizeTrainTest(X_train, X_test):
    scaler = StandardScaler()
    num_cols = ['age', 'creatinine_phosphokinase', 'ejection_fraction',
                'platelets', 'serum_creatinine', 'serum_sodium', 'time']
    # X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    # X_test[num_cols] = scaler.transform(X_test[num_cols])
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    with open('scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)
    return X_train, X_test


def createTrainTest(df, target):
    X = df.drop(target, axis=1)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def predict(data, model):
    y_pred = model.predict(data)
    return y_pred


def predict_proba(values_list, model):
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    # num_cols = ['age', 'creatinine_phosphokinase', 'ejection_fraction',
    #             'platelets', 'serum_creatinine', 'serum_sodium', 'time']
    # header = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
    #       'ejection_fraction', 'high_blood_pressure', 'platelets',
    #       'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']
    # print(len(values_list[0]))
    # print(len(header))
    # data = {header[i]: [values_list[0][i]] for i in range(len(header))}
    # data = pd.DataFrame(data)
    # print(data)
    # data[num_cols] = scaler.transform(data[num_cols])
    data = scaler.transform(values_list)
    # print(data)
    y_pred = model.predict_proba(data)
    # print(y_pred)
    return y_pred


def accuracy(data, realdata):
    accuracy = accuracy_score(realdata, data)
    return accuracy


def save_model(model, acc, model_name):

    # Сериализуем модель в байты
    model_data = pickle.dumps(model)

    save_model_in_db(model_name, acc,model_data)


def choose_best(x_test, y_test, models):
    best = ''
    m = -1
    name=''
    for model in models:
        print(model['name'],end=' ')
        res = accuracy(predict(x_test, model['model']), y_test)
        print(res)
        if (res > m):
            m = res
            best = model['model']
            name = model['name']
    return best,m,name
