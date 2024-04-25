import pandas as pd
import numpy as np
from PIL import Image
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import nltk
import os
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('portuguese'))

mais_stopwords_es = set(["después", "mientras", "luego", "dos", "alli", "toda", "vez"])
mais_stopwords_pt = set(["enquanto", "depois", "após", "onde", "Se", "-se", "-la", "-las", "-los", "-lo", "sobre", "durante", "contra", "antes", "logo", "porém", "Em", "O", "apesar", "então", "dois", "la", "lo", "vez", "em", "o", "através", "agora"])

stop_words = stop_words.union(mais_stopwords_pt)
articles_concat = ""

path = './data/plots_pt'
list_files = os.listdir(path)

for file in list_files:
    if '.txt' not in file:
        continue

    article = open(path + "/" + file, 'r')
    content = article.read()
    
    if content != "":
        articles_concat += content

word_tokens = word_tokenize(articles_concat)

# filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words and w.lower() != 's']
#with no lower case conversion
filtered_sentence = []

 
for w in word_tokens:
    if w.lower() not in stop_words and w.lower() != "'s":
        word = w.lower().split('-')
        filtered_sentence.append(word[0])
 
#print(stop_words)
#print(word_tokens)
print(filtered_sentence)

txt_string = " ".join(filtered_sentence)
#print(txt_string)
wc = WordCloud(background_color='white').generate(txt_string)
plt.imshow(wc)
plt.axis("off")
plt.show()