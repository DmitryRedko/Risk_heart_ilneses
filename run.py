import subprocess # импортирует стандартный модуль subprocess, который позволяет запускать процессы из Python.
cmd = "python main.py" # создает строку cmd, содержащую команду, которая будет выполнена в новом процессе. 
#В данном случае команда запускает скрипт main.py, написанный на языке программирования Python.
p = subprocess.Popen(cmd,shell=False)#  - создает новый процесс, используя команду cmd. 
#Аргумент shell=False указывает, что команда должна быть выполнена без использования 
#командной оболочки операционной системы.
p.communicate() #выполняет команду, созданную ранее, и ожидает ее завершения. 
#Метод communicate() возвращает кортеж, содержащий два элемента: стандартный вывод и 
# стандартный поток ошибок процесса.
