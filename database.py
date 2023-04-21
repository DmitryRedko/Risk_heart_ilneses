import psycopg2
from config import host, user, passwordbd, db_name
from psycopg2 import sql
import psycopg2.extras
import pickle
import pandas as pd
import numpy as np

"""
psycopg2: Эта библиотека является Python-адаптером для работы с PostgreSQL базами данных. 
Она предоставляет функции для подключения к базе данных PostgreSQL, выполнения SQL-запросов, 
обработки результатов запросов и управления транзакциями базы данных.

from config import host, user, passwordbd, db_name: Импортирование переменных host, user, passwordbd, 
db_name из модуля config. Вероятно, эти переменные содержат конфигурационные данные для подключения к 
базе данных PostgreSQL, такие как имя хоста, имя пользователя, пароль и имя базы данных.

psycopg2.sql: Модуль sql в библиотеке psycopg2 предоставляет функции для генерации безопасных SQL-запросов 
с использованием плейсхолдеров, чтобы защитить приложение от атак SQL-инъекций.

psycopg2.extras: Модуль extras в библиотеке psycopg2 содержит дополнительные функции, такие как 
возможность работать с курсорами, возвращающими словари вместо кортежей, и поддержка других типов 
данных PostgreSQL, таких как массивы и диапазоны.

pickle: Модуль pickle в стандартной библиотеке Python предоставляет функции для сериализации и десериализации 
объектов Python, что позволяет сохранять и загружать объекты в/из файлов или передавать их по сети.

pandas: Эта библиотека предоставляет функции для работы с данными в формате таблицы, такими как CSV, Excel, 
SQL, и многими другими. Она также предоставляет удобные функции для манипуляции и анализа данных, такие как 
фильтрация, сортировка, группировка и агрегация данных.

numpy: Эта библиотека предоставляет функции для работы с числовыми массивами и матрицами в Python. Она 
предлагает множество математических функций, таких как операции над массивами, линейная алгебра, статистика 
и т. д. Кроме того, она обладает высокой производительностью и эффективностью, что делает ее идеальным 
инструментом для научных и численных вычислений.
"""

data = pd.read_csv('archive\heart_failure_clinical_records_dataset.csv')

#Подключение к базе данных
conn = psycopg2.connect(
    host=host,
    user=user,
    password=passwordbd,
    database=db_name
)
conn.autocommit = True

"""db_init(): функция, которая создает таблицу patients в базе данных, 
если такой таблицы не существует, и заполняет ее данными из файла CSV. Функция 
выполняется в контексте курсора и не принимает аргументов."""

def db_init():
    with conn.cursor() as cursor:
        cursor.execute('''CREATE TABLE IF NOT EXISTS patients (
                            id SERIAL PRIMARY KEY,
                            age INTEGER,
                            anaemia BOOLEAN,
                            creatinine_phosphokinase INTEGER,
                            diabetes BOOLEAN,
                            ejection_fraction INTEGER,
                            high_blood_pressure BOOLEAN,
                            platelets INTEGER,
                            serum_creatinine REAL,
                            serum_sodium INTEGER,
                            sex REAL,
                            smoking BOOLEAN,
                            time INTEGER,
                            death_event INTEGER
                        )''')
        select_query = "SELECT COUNT(*) FROM patients"
        cursor.execute(select_query)
        result = cursor.fetchone()
        if result[0] == 0:
            for index, row in data.iterrows():
                cursor.execute("INSERT INTO patients (age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, "
                            "high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time, death_event) "
                            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                            (row['age'], bool(row['anaemia']), row['creatinine_phosphokinase'], bool(row['diabetes']), row['ejection_fraction'], bool(row['high_blood_pressure']),
                                row["platelets"], row['serum_creatinine'], row['serum_sodium'], row['sex'], bool(row['smoking']), row['time'], row['DEATH_EVENT']))

"""
insret(): функция, которая вставляет новую запись в таблицу patients. Аргументы функции соответствуют полям таблицы.
"""
def insret(age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure,platelets, serum_creatinine, serum_sodium, sex, smoking, time,death_event):
    with conn.cursor() as cursor:
        cursor.execute("INSERT INTO patients (age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, "
                       "high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time, death_event) "
                       "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                       (age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure,
                        platelets, serum_creatinine, serum_sodium, sex, smoking, time,death_event))


"""
find_helthy_top(name): функция, которая находит параметр name здорового человека. 
Функция выполняет SQL-запрос к базе данных и возвращает значение этого столбца.
"""
def find_helthy_top(name):
    result = ''
    with conn.cursor() as cursor:
        cursor.execute(f"SELECT {name}, COUNT({name}) AS count "
                       "FROM patients "
                       f"GROUP BY {name} "
                       "ORDER BY count DESC "
                       "LIMIT 1;")
        result = cursor.fetchone()
    return result[0]


"""
save_model_in_db(model_name, accuracy, model_data): функция, которая сохраняет обученную модель в базу данных. 
Функция создает таблицу model, если такой таблицы не существует, и вставляет в нее новую запись с именем модели
 model_name, точностью accuracy и данными модели model_data. Для вставки данных модели используется тип данных BYTEA.
"""
def save_model_in_db(model_name,accuracy,model_data):
    with conn.cursor() as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model (
                id SERIAL PRIMARY KEY,
                model_name VARCHAR(255),
                accuracy REAL,
                model_data BYTEA
            )
        """)
        conn.commit()
    
    with conn.cursor() as cursor:
        cursor.execute("""
            INSERT INTO model (model_name, accuracy, model_data)
            VALUES (%s, %s, %s)
        """, (model_name, accuracy, psycopg2.Binary(model_data)))
        conn.commit()


"""
upload_model_from_db(model_name): функция, которая загружает обученную модель из базы данных. Функция выполняет 
SQL-запрос к таблице model, чтобы найти запись с заданным именем модели model_name и загрузить данные модели.
 Если аргумент model_name равен "last", то функция загружает последнюю сохраненную в базе данных модель.
  Функция возвращает загруженную модель и ее точность."""
def upload_model_from_db(model_name):
    with conn.cursor() as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model (
                id SERIAL PRIMARY KEY,
                model_name VARCHAR(255),
                accuracy REAL,
                model_data BYTEA
            )
        """)
        conn.commit()
    if(model_name!="last"):
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
            cursor.execute("""
                SELECT model_data,accuracy 
                FROM model 
                WHERE model_name = %s
            """, (model_name,))

            result = cursor.fetchone()
            if result is not None:
                model_data = result['model_data']
                acc = result['accuracy']
                loaded_model = pickle.loads(model_data)
                print("Модель успешно загружена из базы данных.")
                return loaded_model,acc
            else:
                print("Модель с указанным названием не найдена в базе данных.")
    else:
        with conn.cursor() as cursor:
            # Выполните SQL-запрос для выборки последней записи из таблицы model

            cursor.execute("""
                SELECT model_data,accuracy 
                FROM model 
                ORDER BY id DESC
                LIMIT 1
            """)

            result = cursor.fetchone()
            if result is not None:
                model_data = result[0]
                acc = result[1]
                loaded_model = pickle.loads(model_data)
                print("Модель успешно загружена из базы данных.")
                return loaded_model,acc
            else:
                print("Модель не найдена в базе данных.")



""" Функция по заданному словарю строит SQL запрос в базу данных"""
def get_where(dict_info):
    wh = ''
    columns_checkboxes = ['diabetes_check',
                        'high_blood_pressure_check', 'smoking_check', 'anaemia_check', 'death_event_check']
    for key, value in dict_info.items():
        if value is not None and value!='' and key!='male' and key!='female' and key!='unisex' and key not in columns_checkboxes:
            if value == 'checked' and key!="death_event":
                wh+=str(key) +'='+str(True)+' AND '
            elif key == 'death_event':
                if value == 'checked':
                    wh+=str(key) +'='+str(int(True))+' AND '
                else:
                    wh+=str(key) +'='+str(int(False))+' AND '
            else:
                wh+=str(key) +'='+str(value)+' AND '
    return wh[:-5]

"""
Функция select(info_dict) используется для выполнения запросов на выборку данных из 
таблицы patients базы данных. Функция принимает аргумент info_dict, который содержит 
условия выборки в виде словаря. Функция сначала получает условие выборки с помощью 
функции get_where(), а затем, если условие не пустое, выполняет запрос на выборку данных, 
используя оператор SELECT и условие выборки. Если условие пустое, то выбираются все данные 
из таблицы patients. Результат выборки возвращается в виде списка кортежей.
"""
def select(info_dict):
    cond = get_where(info_dict)
    result = ''
    if(bool(cond)):
        with conn.cursor() as cursor:
            cursor.execute("SELECT * "
                        "FROM patients "
                        f"WHERE {cond}")
            result = cursor.fetchall()
    else:
         with conn.cursor() as cursor:
            cursor.execute("""SELECT * 
                        FROM patients """)
            result = cursor.fetchall()
        
    return result

"""
Функция update_db(info_dict_from, info_dict_to) используется для обновления данных в таблице patients. 
Функция принимает два аргумента: info_dict_from - словарь с информацией об исходных данных для 
обновления и info_dict_to - словарь с новыми данными для обновления. Функция использует оператор UPDATE 
для обновления записей в таблице, сравнивая условия выборки из info_dict_from и заменяя соответствующие
 значения в таблице на значения из info_dict_to.
"""
def update_db(info_dict_from, info_dict_to):
    with conn.cursor() as cursor:
        cursor.execute("UPDATE patients "
                    "SET age = %s, anaemia = %s, creatinine_phosphokinase = %s, diabetes = %s, ejection_fraction = %s, "
                    "high_blood_pressure = %s, platelets = %s, serum_creatinine = %s, serum_sodium = %s, sex = %s, smoking = %s, time =%s, death_event = %s "
                    "WHERE age = %s AND anaemia = %s AND creatinine_phosphokinase = %s AND diabetes = %s AND ejection_fraction = %s AND "
                    "high_blood_pressure = %s AND platelets = %s AND serum_creatinine = %s AND serum_sodium = %s AND sex = %s AND smoking = %s AND time =%s AND death_event = %s ",
                    (info_dict_to['age'], bool(info_dict_to['anaemia']), info_dict_to['creatinine_phosphokinase'], bool(info_dict_to['diabetes']), info_dict_to['ejection_fraction'], bool(info_dict_to['high_blood_pressure']),
                            info_dict_to["platelets"], info_dict_to['serum_creatinine'], info_dict_to['serum_sodium'], info_dict_to['sex'], bool(info_dict_to['smoking']), info_dict_to['time'], int(info_dict_to['death_event']) ,info_dict_from['age'], bool(info_dict_from['anaemia']), info_dict_from['creatinine_phosphokinase'], bool(info_dict_from['diabetes']), info_dict_from['ejection_fraction'], bool(info_dict_from['high_blood_pressure']),
                            info_dict_from["platelets"], info_dict_from['serum_creatinine'], info_dict_from['serum_sodium'], info_dict_from['sex'], bool(info_dict_from['smoking']), info_dict_from['time'], int(info_dict_from['death_event'])))


"""
Функция delete_db(info_dict) используется для удаления записей из таблицы patients. Функция принимает 
аргумент info_dict, содержащий условия удаления в виде словаря. Функция использует оператор DELETE для 
удаления записей из таблицы, сравнивая условия выборки из info_dict.
"""
def delete_db(info_dict):
    with conn.cursor() as cursor:
        cursor.execute("""DELETE FROM patients 
                    WHERE age = %s AND anaemia = %s AND creatinine_phosphokinase = %s AND diabetes = %s AND ejection_fraction = %s AND
                    high_blood_pressure = %s AND platelets = %s AND serum_creatinine = %s AND serum_sodium = %s AND sex = %s AND smoking = %s AND time =%s AND death_event = %s """,
                    (info_dict['age'], bool(info_dict['anaemia']), info_dict['creatinine_phosphokinase'], bool(info_dict['diabetes']), info_dict['ejection_fraction'], bool(info_dict['high_blood_pressure']),
                            info_dict["platelets"], info_dict['serum_creatinine'], info_dict['serum_sodium'], info_dict['sex'], bool(info_dict['smoking']), info_dict['time'], int(info_dict['death_event'])))

"""
Функция get_db() используется для получения всех данных из таблицы patients. Функция использует метод 
read_sql() библиотеки pandas для выполнения запроса на выборку всех данных из таблицы patients и 
загрузки их в объект DataFrame. Затем столбец с идентификаторами записей удаляется из объекта DataFrame.
 В результате, функция возвращает объект DataFrame, содержащий все записи из таблицы patients."""
def get_db():
    df = pd.read_sql("SELECT * FROM patients", conn)
    df.drop(["id"], axis=1, inplace=True)
    return df

