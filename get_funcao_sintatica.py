from transformers import AutoTokenizer, AutoModelForTokenClassification, TokenClassificationPipeline
import re
import os
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def tam_plot(file):

    path = "./data/plots_pt"
    # acessa cada arquivo .html para lê-lo e realizar o parser
    film = path + '/' + file
    
    with open(film, 'r') as fp:
        content = fp.read()
        content = content.replace("==", "")
        content = content.replace("\n", " ")
        list_content = re.split(" ", content)

        for word in list_content:
            if "https://" in word:
                list_content.remove(word)

    tamanho = len(list_content)
    return tamanho



model_name = "QCRI/bert-base-multilingual-cased-pos-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# entidades = {}

path = './data/plots_pt'
text = ""
list_files = os.listdir(path)
atributos = ['N', 'J', 'V']

df = pd.DataFrame(index=list_files, columns=atributos)
df = df.fillna(0)

print(df)

for file in list_files:
    if '.txt' not in file:
        continue

    article = open(path + "/" + file, 'r')
    content = article.read()
    print(file)

    pipeline = TokenClassificationPipeline(model=model, tokenizer=tokenizer)
    text = re.sub(r'[^\w\s]','',content)

    outputs = pipeline(text)

    myset = set()

    for output in outputs:
        if '#' not in output['word']:            
            if output['word'] not in myset:
                myset.add(output['word'])

                if output['entity'] == 'NN' or output['entity'] == 'NNP' or output['entity'] == 'NNS' or output['entity'] == 'NNPS':
                    df.at[file, 'N'] += 1
                elif output['entity'] == 'JJ' or output['entity'] == 'JJS' or output['entity'] == 'JJR':
                    df.at[file, 'J'] += 1
                elif output['entity'] == 'VBD' or output['entity'] == 'VBZ' or output['entity'] == 'VBP' or output['entity'] == 'VB' or output['entity'] == 'VBN' or output['entity'] == 'VBG':
                    df.at[file, 'V'] += 1

                # if output['entity'] in entidades.keys():
                #     entidades[output['entity']] += 1
                # else:
                #     entidades[output['entity']] = 1
    df.loc[file, :] /= tam_plot(file)
# print(df.sum(axis=0))
# print(df.sum(axis=1))
# df = df.div(df.sum(axis=1), axis=0)
df.to_csv("df_pt_funcoes_sint2.csv")
print(df)
# print(entidades)

#entidades_ingles = {'IN': 10449, 'NNP': 13844, 'DT': 3254, 'JJ': 10351, 'NN': 25711, 'VBZ': 12888, 'VBG': 5255, 'PRP$': 1549, 'NNS': 5893, 'CC': 1212, 'JJR': 256, 'VBP': 2632, 'RB': 6107, 'PRP': 3576, 'TO': 614, 'VB': 6477, 'CD': 1699, 'RP': 739, 'RBR': 159, 'VBN': 5112, 'WRB': 942, 'WP': 663, 'MD': 872, 'VBD': 1389, 'JJS': 189, 'RBS': 20, 'WDT': 344, 'NNPS': 327, 'PDT': 49, 'EX': 38, 'WP$': 62, 'FW': 40, 'UH': 4, '$': 1}
#entidades_espanhol = {'IN': 5507, 'CD': 1040, 'DT': 4137, 'NN': 17633, 'JJ': 5023, 'NNP': 10342, 'VBZ': 7548, 'PRP$': 873, 'RB': 3473, 'VB': 3974, 'TO': 887, 'VBN': 2319, 'WDT': 459, 'PRP': 1390, 'CC': 1082, 'NNS': 4542, 'VBP': 1874, 'VBG': 1105, 'WRB': 546, 'VBD': 1249, 'JJR': 75, 'WP': 263, 'MD': 277, 'NNPS': 111, 'RBR': 172, 'JJS': 64, 'RBS': 26, 'PDT': 135, 'RP': 17, 'WP$': 59, 'FW': 15, '$': 1, 'UH': 1, 'EX': 0}
#entidades_portugues = {'IN': 5229, 'CD': 764, 'NNS': 2743, 'NN': 10519, 'DT': 2625, 'NNP': 6368, 'VBZ': 4029, 'TO': 772, 'CC': 792, 'WP': 60, 'PRP': 991, 'RB': 1992, 'VB': 2447, 'VBN': 1466, 'JJ': 2990, 'VBD': 836, 'VBP': 1030, 'RBR': 96, 'WRB': 308, 'VBG': 831, 'MD': 156, 'PRP$': 811, 'JJS': 41, 'WDT': 336, 'PDT': 77, 'JJR': 45, 'NNPS': 102, 'RBS': 25, 'WP$': 26, 'FW': 10, 'RP': 1, '$': 5, 'EX': 0, 'UH': 0}

# entidades_ingles = {'IN': 10449, 'NNP': 13844, 'DT': 3254, 'JJ': 10351, 'NN': 25711, 'V': 33753, 'PRP$': 1549, 'NNS': 5893, 'CC': 1212, 'JJR': 256, 'RB': 6107, 'PRP': 3576, 'TO': 614, 'CD': 1699, 'RP': 739, 'RBR': 159, 'WRB': 942, 'WP': 663, 'MD': 872, 'JJS': 189, 'RBS': 20, 'WDT': 344, 'NNPS': 327, 'PDT': 49, 'EX': 38, 'WP$': 62, 'FW': 40, 'UH': 4, '$': 1}
# entidades_espanhol = {'IN': 5507, 'CD': 1040, 'DT': 4137, 'NN': 17633, 'JJ': 5023, 'NNP': 10342, 'V': 18069, 'PRP$': 873, 'RB': 3473, 'TO': 887, 'WDT': 459, 'PRP': 1390, 'CC': 1082, 'NNS': 4542, 'WRB': 546, 'VBD': 1249, 'JJR': 75, 'WP': 263, 'MD': 277, 'NNPS': 111, 'RBR': 172, 'JJS': 64, 'RBS': 26, 'PDT': 135, 'RP': 17, 'WP$': 59, 'FW': 15, '$': 1, 'UH': 1, 'EX': 0}
# entidades_portugues = {'IN': 5229, 'CD': 764, 'NNS': 2743, 'NN': 10519, 'DT': 2625, 'NNP': 6368, 'V': 10639, 'TO': 772, 'CC': 792, 'WP': 60, 'PRP': 991, 'RB': 1992, 'JJ': 2990, 'RBR': 96, 'WRB': 308, 'MD': 156, 'PRP$': 811, 'JJS': 41, 'WDT': 336, 'PDT': 77, 'JJR': 45, 'NNPS': 102, 'RBS': 25, 'WP$': 26, 'FW': 10, 'RP': 1, '$': 5, 'EX': 0, 'UH': 0}

# entidades_ingles = {'NNP': 13844, 'JJ': 10351, 'NN': 25711, 'V': 33753, 'NNS': 5893, 'JJR': 256, 'JJS': 189, 'NNPS': 327}
# entidades_espanhol = {'NN': 17633, 'JJ': 5023, 'NNP': 10342, 'V': 18069, 'NNS': 4542, 'VBD': 1249, 'JJR': 75, 'NNPS': 111, 'JJS': 64}
# entidades_portugues = {'NNS': 2743, 'NN': 10519, 'NNP': 6368, 'V': 10639, 'JJ': 2990, 'JJS': 41, 'JJR': 45, 'NNPS': 102}

# entidades_ingles = {'Substantivos': 45775, 'Adjetivos': 10796, 'Verbos': 33753}
# entidades_espanhol = {'Substantivos': 32628, 'Adjetivos': 5162, 'Verbos': 18069}
# entidades_portugues = {'Substantivos': 19732, 'Verbos': 10639, 'Adjetivos': 3076}

# total_en = sum(entidades_ingles.values())
# total_es = sum(entidades_espanhol.values())
# total_pt = sum(entidades_portugues.values())

# print(total_en)
# print(total_es)
# print(total_pt)

# data = []
# print(list(entidades_ingles.keys()))
# for key in sorted(list(entidades_ingles.keys())):
#     aux_list = []
#     aux_list.append(key)
#     aux_list.append(entidades_ingles[key] / total_en)
#     aux_list.append(entidades_espanhol[key] / total_es)
#     aux_list.append(entidades_portugues[key] / total_pt)
#     data.append(aux_list)


# df = pd.DataFrame(data, columns=["FUNÇÃO SINTÁTICA", "INGLÊS", "ESPANHOL", "PORTUGUÊS"])
# df.plot(x="FUNÇÃO SINTÁTICA", y=["INGLÊS", "ESPANHOL", "PORTUGUÊS"], kind='bar', figsize=(10,7))
# plt.xticks(rotation="horizontal")
# plt.ylabel("Frequência (%)")
# plt.title("Distribuição das principais função sintáticas nos artigos em cada idioma")
# plt.show()

"""
ranking - inglês
1º) NN
2º) NNP
3º) VBZ
4º) IN
5º) JJ
6º) VB
7º) RB
8º) NNS
9º) VBG
10º) VBN

ranking - espanhol
1º) NN
2º) NNP
3º) VBZ
4º) IN
5º) JJ
6º) NNS
7º) DT
8º) VB
9º) RB
10º) VBN

ranking - português
1º) NN
2º) NNP
3º) IN
4º) VBZ
5º) JJ
6º) NNS
7º) DT
8º) VB
9º) RB
10º) VBN
"""