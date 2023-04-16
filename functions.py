from database import find_helthy_top

def check_for_none(name, val, empty):
    # print(val)
    if val == '' or val is None:
        empty+=1
        return empty,float(find_helthy_top(name))
    else:
        if(val=='male'):
            val=1
        elif(val == 'female'):
            val = 0
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
