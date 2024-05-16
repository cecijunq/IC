import pandas as pd
import os
import re
import numpy as np
from PIL import Image
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords, wordnet
import nltk
from nltk.tokenize import word_tokenize, PunktSentenceTokenizer
from gender_spacy import gender_spacy as gs
import spacy

# https://spacy.io/ tem o parser que consegue detectar pontuação, sem ter que tirar

# GENDER SPACY : biblioteca pt: pt_core_news_sm
# https://spacy.io/models/es

nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
# This will take one argument: the spaCy model you wish to use
# nlp = spacy.load("en_core_web_sm")
nlp = gs.GenderParser("en_core_web_sm")
# nlp.max_lenght = 1600000

mais_stopwords_es = set(["después", "mientras", "luego", "dos", "alli", "toda", "vez"])
mais_stopwords_pt = set(["enquanto", "depois", "após", "onde", "Se", "-se", "-la", "-las", "-los", "-lo", "sobre", "durante", "contra", "antes", "logo", "porém", "Em", "O", "apesar", "então", "dois", "la", "lo", "vez", "em", "o", "através", "agora"])

stop_words = stop_words.union(mais_stopwords_pt)
articles_concat = ""

path = './data/plots_en'
list_files = os.listdir(path)

for file in list_files:
    if '.txt' not in file:
        continue

    article = open(path + "/" + file, 'r')
    content = article.read()
    
    if content != "":
        articles_concat += content

articles_concat = re.sub(r'[^\w\s]','',articles_concat)
print(len(articles_concat))
# doc = nlp.process_doc(articles_concat[:900000])
doc = nlp.process_doc(articles_concat[:900000])

# perform coreference resolution on the doc container
# This part of the library comes from spacy-experimental
doc = nlp.coref_resolution()

# Visualize the result:
nlp.visualize()

word_tokens = word_tokenize(articles_concat, "english")

# filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words and w.lower() != 's']
#with no lower case conversion
filtered_sentence = []

category_words = []
for w in word_tokens:
    # matching: passa o parâmetro 'case' falso OLHAR ISSO
    if w.lower() not in stop_words and w.lower() != "'s":
        word = w.capitalize().split('-')[0]
        # print(nltk.pos_tag(nltk.word_tokenize(word)))
        category_words.append(nltk.pos_tag(nltk.word_tokenize(word))[0])
        # syns = wordnet.synsets(word)
        # if syns:
        #     category_words.append([word, syns[0].lexname().split('.')[0]])
        # print(word, syns[0].lexname().split('.')[0]) if syns else (word, None)
        
        #filtered_sentence.append(word[0])

df = pd.DataFrame(category_words, columns=['Palavra', 'Categoria'])

df = df.groupby('Categoria').count()
print(df)


"""
INGLÊS
Categoria  Palavra      
CC              24
CD            2780
DT             348
FW              25
IN            2748
JJ           10510
JJR            268
JJS            439
LS               7
MD             321
NN           85944
NNP          16795
NNPS            10
NNS          38943
PRP$             7
RB            9549
RBR            145
RBS              8
SYM             18
UH               5
VB            7117
VBD            869
VBG          10818
VBN           8098
WDT              7
WP               8
WRB              8
"""


"""
ESPANHOL
Categoria  Palavra 
CC              10
CD             865
DT              76
EX               1
FW              27
IN             308
JJ            2462
JJR              1
JJS             36
MD              33
NN          107359
NNP          11865
NNS          14340
PRP             35
PRP$            11
RB            1021
SYM             21
TO               8
UH               4
VB            1217
VBD             10
VBG            200
VBN             97
WP               5
WRB             18
"""


"""
PORTUGUÊS
Categoria  Palavra 
CC              15
CD             529
DT              54
FW               4
IN             126
JJ            1037
JJR              2
JJS             35
MD               4
NN           51070
NNP           5141
NNS           6298
PRP             12
RB             250
RBR              2
SYM              7
TO               2
VB             459
VBD              3
VBG             72
VBN             41
WRB              1
"""

#############

"""
INGLÊS
Categoria  Palavra          
CC               5
CD            2894
DT             319
IN            2023
JJ           11731
JJR            188
JJS            204
LS               7
MD             583
NN          107685
NNS          36985
PRP            136
RB            9098
RBR            143
VB            3248
VBD           1234
VBG           9276
VBN           7610
VBZ           2371
WDT              6
WP$             73
"""

"""
ESPANHOL
Categoria  Palavra         
CC               8
CD             867
DT              51
IN              48
JJ            1937
JJS              3
MD              34
NN          122957
NNS          12852
PRP             18
PRP$            11
RB             775
TO               8
VB             115
VBD             42
VBG            161
VBN            121
VBP              7
VBZ              8
WP               5
WRB              2
"""

"""
PORTUGUÊS
Categoria  Palavra         
CC              13
CD             529
DT              35
IN              27
JJ             796
JJS              1
MD               5
NN           57763
NNS           5417
PRP             25
RB             323
RBR              2
TO               2
VB              49
VBD             63
VBG             53
VBN             60
VBP              1
"""

txt_string = " ".join(filtered_sentence)
#print(txt_string)
#wc = WordCloud(background_color='white').generate(txt_string)
#plt.imshow(wc)
#plt.axis("off")
#plt.show()

