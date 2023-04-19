import psycopg2
from config import host, user, passwordbd, db_name
from psycopg2 import sql
import psycopg2.extras
import pickle
import pandas as pd
import numpy as np

def outlier_limits(data, col):
    Q3 = data[col].quantile(0.75)
    Q1 = data[col].quantile(0.25)
    IQR = Q3 - Q1
    UL = Q3 + 1.5*IQR
    LL = Q1 - 1.5*IQR
    return UL, LL


data = pd.read_csv('archive\heart_failure_clinical_records_dataset.csv')
data['age_group'] = np.nan
data.loc[(data['age'] < 65), 'age_group'] = 0
data.loc[(data['age'] >= 65), 'age_group'] = 1

for col in ['creatinine_phosphokinase', 'platelets', 'serum_creatinine']:
    UL, LL = outlier_limits(data, col)
    data[col] = np.where((data[col] > UL), UL, data[col])
    data[col] = np.where((data[col] < LL), LL, data[col])

conn = psycopg2.connect(
    host=host,
    user=user,
    password=passwordbd,
    database=db_name
)
conn.autocommit = True
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


def insret(age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure,platelets, serum_creatinine, serum_sodium, sex, smoking, time):
    with conn.cursor() as cursor:
        cursor.execute("INSERT INTO patients (age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, "
                       "high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time) "
                       "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                       (age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure,
                        platelets, serum_creatinine, serum_sodium, sex, smoking, time))

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

def upload_model_from_db(model_name):
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
                print("Модель с указанным названием не найдена в базе данных.")

def get_whrere(dict_info):
    wh = ''
    # keys = dictionary.keys()
    for key, value in dict_info.items():
        if value is not None and value!='' and key!='male' and key!='female':
            wh+=str(key) +'='+str(value)+' AND '
    print(wh)
    return wh[:-5]



def select(info_dict):
    cond = get_whrere(info_dict)
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