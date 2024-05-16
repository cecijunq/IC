import pandas as pd
import os
import re
import numpy as np
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
nlp = gs.GenderParser("en_core_web_sm")

doc_path = "./data/plots_en/The Godfather.txt"
text = ""

with open(doc_path, 'r') as f:
    text = f.read()

print("X")
doc = nlp.process_doc(text)
print("Y")

# perform coreference resolution on the doc container
# This part of the library comes from spacy-experimental
doc = nlp.coref_resolution()
print("Z")

# Visualize the result:
nlp.visualize()
print("W")


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
      
# nltk.download('stopwords')
# nltk.download('punkt')
# stop_words = set(stopwords.words('english'))
# # This will take one argument: the spaCy model you wish to use
# nlp = spacy.load("en_core_web_sm")

# doc_path = "./data/plots_en/The Godfather.txt"
# text = ""

# with open(doc_path, 'r') as f:
#     text = f.read()

# doc = nlp(text)
# print(doc.text)

# for token in doc:
#     print(token.text, token.pos_, token.dep_)