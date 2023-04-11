from flask import Flask, render_template, request
import psycopg2
from config import host, user, passwordbd, db_name
from LR import Prediction
import pandas as pd
import webbrowser


data = pd.read_csv('archive\heart_failure_clinical_records_dataset.csv')


app = Flask(__name__)

# Конфигурация подключения к базе данных
connection = psycopg2.connect(
    host=host,
    user=user,
    password=passwordbd,
    database=db_name
)

connection.autocommit = True

with connection.cursor() as cursor:
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
                        sex VARCHAR(10),
                        smoking BOOLEAN,
                        time_period INTEGER,
                        death_event INTEGER
                    )''')
    select_query = "SELECT COUNT(*) FROM patients"
    cursor.execute(select_query)
    result = cursor.fetchone()
    if result[0] == 0:
        for index, row in data.iterrows():
            cursor.execute("INSERT INTO patients (age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, "
                           "high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time_period, death_event) "
                           "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                           (row['age'], bool(row['anaemia']), row['creatinine_phosphokinase'], bool(row['diabetes']), row['ejection_fraction'], bool(row['high_blood_pressure']),
                            row["platelets"], row['serum_creatinine'], row['serum_sodium'], row['sex'], bool(row['smoking']), row['time'], row['DEATH_EVENT']))


# Открытие веб-страницы
@app.route('/')
def home():
    return render_template('index.html', data={'accuracy': '', 'probability': ''})

# Добавление данных в базу данных


def find_helthy_top(name):
    result = ''
    with connection.cursor() as cursor:
        cursor.execute(f"SELECT {name}, COUNT({name}) AS count "
                       "FROM patients "
                       f"GROUP BY {name} "
                       "ORDER BY count DESC "
                       "LIMIT 1;")
        result = cursor.fetchone()
    return result[0]


def check_for_none(name, val):
    # print(val)
    k=0
    if val == '' or val is None:
        k=1
        return k,find_helthy_top(name)
    else:
        return k,val


@app.route('/add_post', methods=['POST', 'GET'])
def add_data():
    # Получение данных из формы на веб-странице
    age = request.form.get('age_add')
    anaemia = request.form.get('anaemia_add')
    creatinine_phosphokinase = request.form.get('creatinine_phosphokinase_add')
    diabetes = request.form.get('diabetes_add')
    ejection_fraction = request.form.get('ejection_fraction_add')
    high_blood_pressure = request.form.get('high_blood_pressure_add')
    platelets = request.form.get('platelets_add')
    serum_creatinine = request.form.get('serum_creatinine_add')
    serum_sodium = request.form.get('serum_sodium_add')
    sex = request.form.get('sex_add')
    smoking = request.form.get('smoking_add')
    time_period = request.form.get('time_period_add')

    with connection.cursor() as cursor:
        cursor.execute("INSERT INTO patients (age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, "
                       "high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time_period) "
                       "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                       (age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure,
                        platelets, serum_creatinine, serum_sodium, sex, smoking, time_period))

    print('Данные успешно добавлены в базу данных!')
    return render_template('index.html', data={'accuracy': '', 'probability': ''})


@app.route('/prediction', methods=['POST', 'GET'])
def prediction():
    # Получение данных из формы на веб-странице
    count_empty_fields_for_prediction = 0

    k, age = check_for_none('age', request.form.get('age'))
    count_empty_fields_for_prediction+=k
    k, anaemia = check_for_none('anaemia', request.form.get('anaemia'))
    count_empty_fields_for_prediction+=k
    k,creatinine_phosphokinase = check_for_none('creatinine_phosphokinase', request.form.get('creatinine_phosphokinase'))
    count_empty_fields_for_prediction+=k
    k,diabetes = check_for_none('diabetes', request.form.get('diabetes'))
    count_empty_fields_for_prediction+=k
    k,ejection_fraction = check_for_none('ejection_fraction', request.form.get('ejection_fraction'))
    count_empty_fields_for_prediction+=k
    k,high_blood_pressure = check_for_none('high_blood_pressure', request.form.get('high_blood_pressure'))
    count_empty_fields_for_prediction+=k
    k,platelets = check_for_none('platelets', request.form.get('platelets'))
    count_empty_fields_for_prediction+=k
    k,serum_creatinine = check_for_none('serum_creatinine', request.form.get('serum_creatinine'))
    count_empty_fields_for_prediction+=k
    k,serum_sodium = check_for_none('serum_sodium', request.form.get('serum_sodium'))
    count_empty_fields_for_prediction+=k
    k,sex = check_for_none('sex', request.form.get('sex'))
    count_empty_fields_for_prediction+=k
    k, smoking = check_for_none('smoking', request.form.get('smoking'))
    k,time_period = check_for_none('time_period', request.form.get('time_period'))
    count_empty_fields_for_prediction+=k

    # print("COUNT:",count_empty_fields_for_prediction)
    age = float(age)
    anaemia = int(bool(anaemia))
    creatinine_phosphokinase = float(creatinine_phosphokinase)
    diabetes = int(bool(diabetes))
    ejection_fraction = float(ejection_fraction)
    high_blood_pressure = int(bool(high_blood_pressure))
    platelets = float(platelets)
    serum_creatinine = float(serum_creatinine)
    serum_sodium = float(serum_sodium)
    sex = int(bool(sex))
    smoking = int(bool(smoking))
    time_period = float(time_period)

    # print(age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction,
        #   high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time_period)

    model = Prediction()
    X_train, X_test, y_train, y_test = model.createTrainTest(
        data, "DEATH_EVENT")
    X_train, X_test = model.NormalaizeTrainTest(X_train, X_test)
    mod = model.LR(X_train, y_train)
    y_pred = model.predict(X_test, mod)
    acc = model.accuracy(y_pred, y_test)
    prob = model.predict_proba([[age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction,
                                 high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time_period]], mod)

    print(acc, prob)
    return render_template('index.html', data={'accuracy': round(acc*100, 3), 'probability': round(prob[0][1]*100, 3), 'AccurParam': round(100-(11-count_empty_fields_for_prediction)/11*100)})


if __name__ == '__main__':
    webbrowser.open_new('http://127.0.0.1:2000/')
    app.run(port=2000)
