import subprocess
cmd = "python main.py"
p = subprocess.Popen(cmd,shell=False)
p.communicate()
