import pandas as pd
import os
import re

path = './data/plots_en'
list_files = os.listdir(path)
# list_files = ["12 Angry Men (1957 film).txt"]

articles = pd.DataFrame()
pattern = re.compile(r'== Plot|Synopsis ==\n(.*?)\n== (\S+) ==', re.DOTALL)

for file in list_files:
    if '.txt' not in file:
        continue
    # print(file)

    article = open(path + "/" + file, 'r')
    new_line = file.split(".txt")[0]
    content = article.read()
    
    if content != "":
        with open(path + "/" + file, 'r+') as file: 
            file.seek(0, 0) 
            file.write(new_line + '\n' + content) 