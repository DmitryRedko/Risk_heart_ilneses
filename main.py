from flask import Flask, render_template, request, url_for 
"""Эта строка импортирует веб-фреймворк Flask, а также некоторые конкретные функции 
(render_template, request и url_for), которые будут использоваться для рендеринга HTML-шаблонов, 
обработки запросов пользователей и генерации URL-адресов для различных частей приложения."""

from mlModels import Models, createTrainTest, NormalaizeTrainTest, predict, predict_proba, accuracy, save_model, choose_best
"""Эта строка импортирует различные функции из модуля mlModels, которые будут использоваться 
для работы с моделями машинного обучения, включая создание обучающего и тестового наборов данных,
 нормализацию данных, прогнозирование, вычисление точности модели, сохранение модели и выбор лучшей модели."""

import pandas as pd
"""Эта строка импортирует модуль pandas, который используется для работы с данными, такими 
как чтение и запись CSV-файлов, работа с таблицами данных и преобразование данных."""

from database import db_init
"""Эта строка импортирует функцию db_init из модуля database, которая будет использоваться для инициализации базы данных."""

from functions import check_for_none, get_status, convert_to_normal, do_prediction, add_to_db, action_select,action_update, action_delete,reeducate
"""Эта строка импортирует различные функции из модуля functions, которые будут использоваться для обработки 
запросов от пользователей, включая проверку наличия данных, получение статуса, преобразование данных, выполнение 
прогнозирования, добавление данных в базу данных, выполнение запросов на выборку, обновление и удаление 
данных и переобучение модели."""

import webbrowser
"""Эта строка импортирует модуль webbrowser, который позволяет открывать веб-страницы в браузере пользователя."""

import numpy as np
"""Эта строка импортирует модуль numpy, который используется для работы с массивами и матрицами, а также для выполнения вычислений."""

import warnings
"""Эта строка импортирует модуль warnings, который позволяет выводить предупреждения при выполнении кода."""

warnings.filterwarnings("ignore")
"""вызывает функцию filterwarnings из модуля warnings, чтобы отключить 
вывод предупреждений во время выполнения программы."""

data = pd.read_csv('archive\heart_failure_clinical_records_dataset.csv')

"""- использует метод read_csv() из библиотеки pandas для чтения данных из CSV-файла 
heart_failure_clinical_records_dataset.csv, находящегося в папке archive. В результате 
данных будет создан объект DataFrame, который сохраняется в переменной data."""

app = Flask(__name__)
"""
Cоздает экземпляр класса Flask из библиотеки flask. 
Этот объект будет использоваться для создания веб-приложения.
"""
db_init()

""" вызывает функцию db_init(), которая инициализирует базу данных."""

"""@app.route('/') - декоратор, который определяет маршрут по умолчанию. При запросе этого маршрута 
в браузере будет возвращен результат функции home(). Эта функция рендерит HTML-шаблон predict.html с 
пустым словарем dataPredict"""
@app.route('/')
def home():
    return render_template('predict.html', dataPredict={})

"""@app.route('/add', methods=['POST', 'GET']) - определяет маршрут /add. 
Когда этот маршрут запрашивается, то происходит вызов функции add_data(). 
Выполняются определенные действия в зависимости от нажатой кнопки на веб-странице. 
Функция add_to_db() добавляет данные в базу данных и выводит сообщение об
успешном добавлении или об ошибке при нажатии добавить в базу.
Функция reeducate() обучает модель с учетом добавленной строки при нажатии обучить.
"""
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

"""@app.route('/prediction', methods=['POST', 'GET']) - определяет маршрут /prediction. 
При запросе этого маршрута в браузере будет возвращен результат функции prediction(). 
Если метод запроса - POST, то происходит вызов функции do_prediction(), которая проводит 
прогнозирование с использованием модели машинного обучения. Функция prediction() словарь
для рендеринга HTML-шаблона predict.html с результатом прогнозирования."""

@app.route('/prediction', methods=['POST', 'GET'])
def prediction():
    info_dict=''
    if "predict_button" in request.form:
        try:
            info_dict = do_prediction(request)
        except:
            print("Ошибка прогнозирования, попробуйте обучить модель заново.")
    return render_template('predict.html',  dataPredict=info_dict)


"""
@app.route('/update', methods=['POST', 'GET']) - определяет маршрут /update. 
Когда этот маршрут запрашивается, то происходит вызов функции update(). Если метод 
запроса - POST, то выполняются определенные действия в зависимости от нажатой кнопки 
на веб-странице (вывод базы данных или обновление строки).
Функция update() выполняет обновление данных в базе данных и выводит 
сообщение об успешном обновлении или об ошибке.
"""
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

"""
@app.route('/delete', methods=['POST', 'GET']) - определяет маршрут /delete. 
Когда этот маршрут запрашивается, то происходит вызов функции delete(). Если метод 
запроса - POST, то выполняются определенные действия в зависимости от нажатой кнопки 
на веб-странице(вывод базы данных или удаление строки). Функция delete() выполняет 
удаление данных из базы данных и выводит сообщение об успешном удалении или об ошибке.
"""

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

"""
if __name__ == '__main__': - эта строка проверяет, является ли данный файл исполняемым.
Если да, то выполняется блок кода, который открывает веб-страницу в браузере и запускает 
приложение с указанным портом 2000.
"""
if __name__ == '__main__':
    webbrowser.open_new('http://127.0.0.1:2000/')
    app.run(port=2000)
