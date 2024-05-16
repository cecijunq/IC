# LINK REFERÊNCIA: https://huggingface.co/Babelscape/wikineural-multilingual-ner

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import os
import re
import pandas as pd
import matplotlib.pyplot as plt

n_per = 0
n_org = 0
n_loc = 0
n_misc = 0

values_per = ['PER']
values_org = ['ORG']
values_loc = ['LOC']
values_misc = ['MISC']

tokenizer = AutoTokenizer.from_pretrained("Babelscape/wikineural-multilingual-ner")
model = AutoModelForTokenClassification.from_pretrained("Babelscape/wikineural-multilingual-ner")

path = './data/plots_en'
text = ""
list_files = os.listdir(path)

for file in list_files:
    if '.txt' not in file:
        continue

    article = open(path + "/" + file, 'r')
    content = article.read()

    content = re.sub(r'[^\w\s]','',content)
    nlp = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)

    ner_results = nlp(content)

    for element in ner_results:
        if element['entity_group'] == 'PER':
            n_per += 1
        elif element['entity_group'] == 'ORG':
            n_org += 1
        if element['entity_group'] == 'LOC':
            n_loc += 1
        if element['entity_group'] == 'MISC':
            n_misc += 1

total = n_per + n_org + n_loc + n_misc
values_per.append(n_per / total)
values_org.append(n_org / total)
values_loc.append(n_loc / total)
values_misc.append(n_misc / total)

########

n_per = 0
n_org = 0
n_loc = 0
n_misc = 0

path = './data/plots_es'
text = ""
list_files = os.listdir(path)

for file in list_files:
    if '.txt' not in file:
        continue

    article = open(path + "/" + file, 'r')
    content = article.read()

    content = re.sub(r'[^\w\s]','',content)
    nlp = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)

    ner_results = nlp(content)

    for element in ner_results:
        if element['entity_group'] == 'PER':
            n_per += 1
        elif element['entity_group'] == 'ORG':
            n_org += 1
        if element['entity_group'] == 'LOC':
            n_loc += 1
        if element['entity_group'] == 'MISC':
            n_misc += 1

total = n_per + n_org + n_loc + n_misc
values_per.append(n_per / total)
values_org.append(n_org / total)
values_loc.append(n_loc / total)
values_misc.append(n_misc / total)

########

n_per = 0
n_org = 0
n_loc = 0
n_misc = 0

path = './data/plots_pt'
text = ""
list_files = os.listdir(path)

for file in list_files:
    if '.txt' not in file:
        continue

    article = open(path + "/" + file, 'r')
    content = article.read()

    content = re.sub(r'[^\w\s]','',content)
    nlp = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)

    ner_results = nlp(content)

    for element in ner_results:
        if element['entity_group'] == 'PER':
            n_per += 1
        elif element['entity_group'] == 'ORG':
            n_org += 1
        if element['entity_group'] == 'LOC':
            n_loc += 1
        if element['entity_group'] == 'MISC':
            n_misc += 1

total = n_per + n_org + n_loc + n_misc
values_per.append(n_per / total)
values_org.append(n_org / total)
values_loc.append(n_loc / total)
values_misc.append(n_misc / total)


# print(n_per, n_org, n_loc, n_misc)

# values_en = ['Inglês', 19242, 928, 2331, 4086]
# values_es = ['Espanhol', 8554, 406, 1396, 941]
# values_pt = ['Português', 5059, 198, 1011, 638]

# values_per = ['PER', 19242, 8554, 5059]
# values_org = ['ORG', 928, 406, 198]
# values_loc = ['LOC', 2332, 1396, 1011]
# values_misc = ['MISC', 4086, 941, 638]

data = [values_per, values_org, values_loc, values_misc]

df = pd.DataFrame(data, columns=["ENTIDADE", "INGLÊS", "ESPANHOL", "PORTUGUÊS"])
df.plot(x="ENTIDADE", y=["INGLÊS", "ESPANHOL", "PORTUGUÊS"], kind='bar', figsize=(10,10))
plt.title("Distribuição da presença de entidades nos artigos de cada idioma")
plt.show()
