from flask import Flask, render_template, request
from mlModels import Models, createTrainTest, NormalaizeTrainTest, predict, predict_proba, accuracy, save_model, choose_best
import pandas as pd
from database import db_init, insret, find_helthy_top, upload_model_from_db
from functions import check_for_none, get_status
import webbrowser
import numpy as np
import warnings

# Отключение предупреждений
warnings.filterwarnings("ignore")


# def outlier_limits(data, col):
#     Q3 = data[col].quantile(0.75)
#     Q1 = data[col].quantile(0.25)
#     IQR = Q3 - Q1
#     UL = Q3 + 1.5*IQR
#     LL = Q1 - 1.5*IQR
#     return UL, LL


data = pd.read_csv('archive\heart_failure_clinical_records_dataset.csv')
# data['age_group'] = np.nan
# data.loc[(data['age'] < 65), 'age_group'] = 0
# data.loc[(data['age'] >= 65), 'age_group'] = 1

# for col in ['creatinine_phosphokinase', 'platelets', 'serum_creatinine']:
#     UL, LL = outlier_limits(data, col)
#     data[col] = np.where((data[col] > UL), UL, data[col])
#     data[col] = np.where((data[col] < LL), LL, data[col])

app = Flask(__name__)

db_init()

# Открытие веб-страницы


@app.route('/')
def home():
    return render_template('index.html', data={})


@app.route('/add_post', methods=['POST', 'GET'])
def add_data():
    # Получение данных из формы на веб-странице
    columns = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 'high_blood_pressure', 'platelets',
               'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time', 'death_event']
    info_dict = {}
    for name in columns:
        temp = request.form.get(name)
        if (temp == 'male'):
            temp = 0
        elif (temp == 'female'):
            temp = 1
        info_dict[name] = temp

    insret(info_dict['age'], info_dict['anaemia'], info_dict['creatinine_phosphokinase'], info_dict['diabetes'], info_dict['ejection_fraction'],
           info_dict['high_blood_pressure'], info_dict['platelets'], info_dict['serum_creatinine'], info_dict['serum_sodium'], info_dict['sex'], info_dict['smoking'], info_dict['time'])

    print('Данные успешно добавлены в базу данных!')
    return render_template('index.html', data={})


@app.route('/prediction', methods=['POST', 'GET'])
def prediction():
    # Получение данных из формы на веб-странице
    columns = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets',
               'serum_creatinine', 'serum_sodium', 'sex', 'time']
    columns_checkboxes = ['diabetes',
                          'high_blood_pressure', 'smoking', 'anaemia']
    info_dict = {}
    empty_field = 0
    for name in columns:
        empty_field, info_dict[name] = check_for_none(
            name, request.form.get(name), empty_field)

    missing_checkboxes = 0
    fictval = 0
    for name in columns_checkboxes:
        if bool(request.form.get(name+'_check')) == 0:
            info_dict[name] = int(bool(request.form.get(name)))
        else:
            missing_checkboxes += 1
            fictval, info_dict[name] = check_for_none(
                name, request.form.get(name), fictval)
            info_dict[name] = int(bool(name))

    X_train, X_test, y_train, y_test = createTrainTest(data, "DEATH_EVENT")
    X_train, X_test = NormalaizeTrainTest(X_train, X_test)
    models = [Models().RFC(X_train, y_train),Models().KNN(X_train, y_train), Models().LR(X_train, y_train), Models().KernelSVM(
        X_train, y_train), Models().NBayes(X_train, y_train), Models().tree(X_train, y_train)]
    model, acc, model_name = choose_best(X_test, y_test, models)
    save_model(model, acc, model_name)
    # print("--------------------",model_name,"----------------------------")
    uploaded,acc = upload_model_from_db('last')

    prob = predict_proba([[info_dict['age'], info_dict['anaemia'], info_dict['creatinine_phosphokinase'], info_dict['diabetes'], info_dict['ejection_fraction'],
                           info_dict['high_blood_pressure'], info_dict['platelets'], info_dict['serum_creatinine'], info_dict['serum_sodium'], info_dict['sex'], info_dict['smoking'], info_dict['time']]], uploaded)

    return render_template('index.html', data={'status': f'({get_status(prob[0][1])})', 'accuracy': f'{round(acc*100, 3)} %', 'probability': f'{round(prob[0][1]*100, 3)} %', 'AccurParam': f'{round(100-(12 - missing_checkboxes - empty_field)/12*100)} %'})


if __name__ == '__main__':
    webbrowser.open_new('http://127.0.0.1:2000/')
    app.run(port=2000)
