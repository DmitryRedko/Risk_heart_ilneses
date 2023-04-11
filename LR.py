import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

class Prediction:
    def LR(self,X_train,y_train):
        self.lr = LogisticRegression(max_iter=100000)
        param_grid = {
            'C': [0.0001,0.001, 0.01, 0.1, 1.0 ,2.0,3.0, 10, 50, 100, 1000, 10000]
        }
        grid_search = GridSearchCV(self.lr, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        best_lr = grid_search.best_estimator_     
        return best_lr   

    def createTrainTest(self,df,target):
        X = df.drop(target, axis=1)
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def NormalaizeTrainTest(self, X_train,X_test):
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train.values)
        X_test = (self.scaler).transform(X_test.values)
        return X_train, X_test


    def predict(self,data,model):
        y_pred = model.predict(data)
        return y_pred

    def predict_proba(self,data,model):
        X_new = self.scaler.transform(data)
        y_pred = model.predict_proba(X_new)
        print(y_pred)
        return y_pred

    def accuracy(self,data,realdata):
        accuracy = accuracy_score(realdata, data)
        return accuracy

