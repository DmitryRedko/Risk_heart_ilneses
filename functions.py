
from mlModels import Models, createTrainTest, NormalaizeTrainTest, predict, predict_proba, accuracy, save_model, choose_best
import pandas as pd
from database import db_init, insret, find_helthy_top, upload_model_from_db, select, update_db,delete_db, get_db


def check_for_none(name, val, empty):
    if val == '' or val is None:
        empty+=1
        return empty,float(find_helthy_top(name))
    else:
        if(val=='male'):
            val=0
        elif(val == 'female'):
            val = 1
        elif(val == 'on'):
            val = 1
        elif(val == 'off'):
            val = 0
        return empty,float(val)

def get_status(x):
    if(x<=0.33):
        return "Низкая"
    elif(x<=0.67):
        return "Средняя"
    else:
        return "Высокая"

def convert_to_normal(dict_info):
    columns = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets',
               'serum_creatinine', 'serum_sodium', 'sex', 'time','diabetes',
                          'high_blood_pressure', 'smoking', 'anaemia','death_event']
    new_dict={}
    for key, value in dict_info.items():
        for col in columns:
            if col in key:
                new_dict[col] = dict_info[key]
    return new_dict

def do_prediction(request):
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

    uploaded, acc = upload_model_from_db('last')

    prob = predict_proba([[info_dict['age'], info_dict['anaemia'], info_dict['creatinine_phosphokinase'], info_dict['diabetes'], info_dict['ejection_fraction'],
                           info_dict['high_blood_pressure'], info_dict['platelets'], info_dict['serum_creatinine'], info_dict['serum_sodium'], info_dict['sex'], info_dict['smoking'], info_dict['time']]], uploaded)

    for name in columns_checkboxes:
        if bool(request.form.get(name+'_check')) == 0:
            if (bool(info_dict[name])):
                info_dict[name] = "checked"
        else:
            info_dict[name+'_check'] = 'checked'
    info_dict['female'] = 'checked' if info_dict['sex'] == 1 else ''
    info_dict['male'] = 'checked' if info_dict['sex'] == 0 else ''
    info_dict['status'] = f'({get_status(prob[0][1])})'
    info_dict['accuracy'] = f'{round(acc*100, 3)} %'
    info_dict['probability'] = f'{round(prob[0][1]*100, 3)} %'
    info_dict['AccurParam'] = f'{round(100-(12 - missing_checkboxes - empty_field)/12*100)} %'

    return info_dict

def reeducate():
    data = get_db()
    print(data)
    data = data.dropna()
    X_train, X_test, y_train, y_test = createTrainTest(data, "death_event")
    X_train, X_test = NormalaizeTrainTest(X_train, X_test)
    models = [Models().RFC(X_train, y_train), Models().KNN(X_train, y_train), Models().LR(X_train, y_train), Models().KernelSVM(
        X_train, y_train), Models().NBayes(X_train, y_train), Models().tree(X_train, y_train),Models().RFR(X_train, y_train)]
    model, acc, model_name = choose_best(X_test, y_test, models)
    save_model(model, acc, model_name)
    print("Saved model: ",model_name)

def add_to_db(request):
        # Получение данных из формы на веб-странице
    columns = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 'high_blood_pressure', 'platelets',
               'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time', 'death_event']
    columns_checkboxes = ['diabetes',
                          'high_blood_pressure', 'smoking', 'anaemia']
    info_dict = {}
    for name in columns:
        temp = request.form.get(name)
        if (temp == 'male'):
            temp = 0
        elif (temp == 'female'):
            temp = 1
        elif (temp == 'on'):
            temp = 1
        elif (temp is None and name in columns_checkboxes):
            temp = 0
        info_dict[name] = temp

    for key, value in info_dict.items():
        if key is None or value is None or value == '':
            raise ValueError()

    insret(info_dict['age'], bool(info_dict['anaemia']), info_dict['creatinine_phosphokinase'], bool(info_dict['diabetes']), info_dict['ejection_fraction'],
           bool(info_dict['high_blood_pressure']), info_dict['platelets'], info_dict['serum_creatinine'], info_dict['serum_sodium'], info_dict['sex'], bool(info_dict['smoking']), info_dict['time'],info_dict['death_event'])

def action_update(request):
    columns_from = ['age_from', 'creatinine_phosphokinase_from', 'ejection_fraction_from', 'platelets_from',
                    "serum_creatinine_from", 'serum_sodium_from', 'sex_from', 'time_from', '']
    columns_checkboxes_from = ['diabetes_from',
                               'high_blood_pressure_from', 'smoking_from', 'anaemia_from', 'death_event_from']
    info_dict_from = {}
    for name in columns_from:
        temp = request.form.get(name)
        if (temp == 'male'):
            info_dict_from["male_from"] = "checked"
            temp = 0.0
        elif (temp == 'female'):
            info_dict_from["female_from"] = "checked"
            temp = 1.0
        info_dict_from[name] = temp

    for name in columns_checkboxes_from:
        if bool(request.form.get(name+'_check')) == 0:
            if name != "death_event_from":
                info_dict_from[name] = bool(request.form.get(name))
            else:
                info_dict_from[name] = int(bool(request.form.get(name)))
        else:
            info_dict_from[name] = ''

    for name in columns_checkboxes_from:
        if bool(request.form.get(name+'_check')) == 0:
            if (bool(info_dict_from[name])):
                info_dict_from[name] = "checked"
        else:
            info_dict_from[name+'_check'] = 'checked'

    columns_to = ['age_to', 'creatinine_phosphokinase_to', 'ejection_fraction_to', 'platelets_to',
                  'serum_creatinine_to', 'serum_sodium_to', 'sex_to', 'time_to']
    columns_checkboxes_to = ['diabetes_to',
                             'high_blood_pressure_to', 'smoking_to', 'anaemia_to', 'death_event_to']

    info_dict_to = {}
    for name in columns_to:
        temp = request.form.get(name)
        if (temp == 'male'):
            info_dict_to["male_to"] = "checked"
            temp = 0.0
        elif (temp == 'female'):
            info_dict_to["female_to"] = "checked"
            temp = 1.0
        info_dict_to[name] = temp

    for name in columns_checkboxes_to:
        if bool(request.form.get(name+'_check')) == 0:
            if name != "death_event_to":
                info_dict_to[name] = bool(request.form.get(name))
            else:
                info_dict_to[name] = int(bool(request.form.get(name)))
        else:
            info_dict_to[name] = ''

    for name in columns_checkboxes_to:
        if bool(request.form.get(name+'_check')) == 0:
            if (bool(info_dict_to[name])):
                info_dict_to[name] = "checked"
        else:
            info_dict_to[name+'_check'] = 'checked'

    try:
        update_db(convert_to_normal(info_dict_from),
                  convert_to_normal(info_dict_to))
        print("Данные успешно обновлены")
    except:
        print("Ошибка обновления данных")
    return info_dict_from, info_dict_to


def action_select(request):
    columns = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets',
               'serum_creatinine', 'serum_sodium', 'sex', 'time']
    columns_checkboxes = ['diabetes',
                          'high_blood_pressure', 'smoking', 'anaemia', 'death_event']
    info_dict = {}
    for name in columns:
        temp = request.form.get(name)
        if (temp == 'male'):
            info_dict["male"] = "checked"
            temp = 0.0
        elif (temp == 'female'):
            info_dict["female"] = "checked"
            temp = 1.0
        elif (temp == 'unisex'):
            info_dict["unisex"] = "checked"
            temp = ''
        info_dict[name] = temp

    for name in columns_checkboxes:
        if bool(request.form.get(name+'_check')) == 0:
            if name != "death_event":
                info_dict[name] = bool(request.form.get(name))
            else:
                info_dict[name] = int(bool(request.form.get(name)))
        else:
            info_dict[name] = ''
    for name in columns_checkboxes:
        if bool(request.form.get(name+'_check')) == 0:
            if (bool(info_dict[name])):
                info_dict[name] = "checked"
        else:
            info_dict[name+'_check'] = 'checked'

    database_selected = select(info_dict)
    
    for name in columns_checkboxes:
        if bool(request.form.get(name+'_check')) == 0:
            if (bool(info_dict[name])):
                info_dict[name] = "checked"
        else:
            info_dict[name+'_check'] = 'checked'

    return database_selected, info_dict

def action_delete(request):
    columns_delete = ['age_delete', 'creatinine_phosphokinase_delete', 'ejection_fraction_delete', 'platelets_delete',
                    "serum_creatinine_delete", 'serum_sodium_delete', 'sex_delete', 'time_delete', '']
    columns_checkboxes_delete = ['diabetes_delete',
                               'high_blood_pressure_delete', 'smoking_delete', 'anaemia_delete', 'death_event_delete']
    info_dict_delete = {}
    for name in columns_delete:
        temp = request.form.get(name)
        if (temp == 'male'):
            info_dict_delete["male_delete"] = "checked"
            temp = 0.0
        elif (temp == 'female'):
            info_dict_delete["female_delete"] = "checked"
            temp = 1.0
        info_dict_delete[name] = temp

    for name in columns_checkboxes_delete:
        if bool(request.form.get(name+'_check')) == 0:
            if name != "death_event_delete":
                info_dict_delete[name] = bool(request.form.get(name))
            else:
                info_dict_delete[name] = int(bool(request.form.get(name)))
        else:
            info_dict_delete[name] = ''

    for name in columns_checkboxes_delete:
        if bool(request.form.get(name+'_check')) == 0:
            if (bool(info_dict_delete[name])):
                info_dict_delete[name] = "checked"
        else:
            info_dict_delete[name+'_check'] = 'checked'
    delete_db(convert_to_normal(info_dict_delete))