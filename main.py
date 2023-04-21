from flask import Flask, render_template, request, url_for
from mlModels import Models, createTrainTest, NormalaizeTrainTest, predict, predict_proba, accuracy, save_model, choose_best
import pandas as pd
from database import db_init
from functions import check_for_none, get_status, convert_to_normal, do_prediction, add_to_db, action_select,action_update, action_delete,reeducate
import webbrowser
import numpy as np
import warnings

# Отключение предупреждений
warnings.filterwarnings("ignore")

data = pd.read_csv('archive\heart_failure_clinical_records_dataset.csv')

app = Flask(__name__)

db_init()

# Открытие веб-страницы


@app.route('/')
def home():
    return render_template('predict.html', dataPredict={}, dataAdd={}, dataUpdate={}, dataDelete={})


@app.route('/add', methods=['POST', 'GET'])
def add_data():
    if "add_button" in request.form:
        try:
                add_to_db(request)
                print('Данные успешно добавлены в базу данных!')
        except:
            print('Ошибка добавления!')
        
    if "reeducate_button" in request.form:
        try:
            reeducate()
            print('Модели обучены на новых данных!')
        except:
            print('Ошибка обучения!')
    return render_template('add.html',dataAdd={})


@app.route('/prediction', methods=['POST', 'GET'])
def prediction():
    info_dict=''
    if "predict_button" in request.form:
        info_dict = do_prediction(request)
    return render_template('predict.html',  dataPredict=info_dict)

@app.route('/update', methods=['POST', 'GET'])
def update():
    info_dict={}
    info_dict_from={}
    info_dict_to={}
    db_info = ''
    if "select_button" in request.form:
        db_info, info_dict = action_select(request)
    
    if "update_button" in request.form:
        info_dict_from, info_dict_to = action_update(request)
        db_info, info_dict = action_select(request)

    return render_template('update.html', dataUpdate=info_dict | info_dict_to | info_dict_from, db=db_info)

@app.route('/delete', methods=['POST', 'GET'])
def delete():
    info_dict={}
    info_dict_delete={}
    db_info = ''
    
    if "select_button" in request.form:
        db_info, info_dict = action_select(request)
    
    if "delete_button" in request.form:
        # info_dict_from, info_dict_to = action_delete(request)
        action_delete(request)
        db_info, info_dict = action_select(request)

    return render_template('delete.html', dataDelete =  info_dict | info_dict_delete, db=db_info)


if __name__ == '__main__':
    webbrowser.open_new('http://127.0.0.1:2000/')
    app.run(port=2000)
