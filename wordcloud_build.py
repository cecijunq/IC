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
stop_words = set(stopwords.words('english'))

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

word_tokens = word_tokenize(articles_concat)

# filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words and w.lower() != 's']
#with no lower case conversion
filtered_sentence = []

 
for w in word_tokens:
    if w.lower() not in stop_words and w.lower() != "'s":
        filtered_sentence.append(w)
 
#print(stop_words)
#print(word_tokens)
print(filtered_sentence)

txt_string = " ".join(filtered_sentence)
#print(txt_string)
wc = WordCloud(background_color='white').generate(txt_string)
plt.imshow(wc)
plt.axis("off")
plt.show()