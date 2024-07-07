import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import liwc
# from liwc import LIWC
import os
import re
from collections import Counter

# https://pypi.org/project/liwc/

plots = pd.read_csv("./oscar_full_plot.csv", sep=';')

def set_keys(key):
    #['std linguistic dim', 'social processes', 'affective processes', 'cognitive processes', 'perceptual processes', 'biological processes', 'relativity', 'personal concerns', 'spoken categories']
    if key == 'social':
        return 'social processes'
    elif key == 'relativ':
        return 'std linguistic dim'
    elif key == 'verb':
        return 'std linguistic dim'
    elif key == 'pronoun':
        return 'std linguistic dim'
    elif key == 'article':
        return 'std linguistic dim'
    elif key == 'ppron':
        return 'std linguistic dim'
    elif key == 'space':
        return 'relativity'
    elif key == 'conj':
        return 'std linguistic dim'
    elif key == 'shehe':
        return 'std linguistic dim'
    elif key == 'affect':
        return 'affective processes'
    elif key == 'auxverb':
        return 'std linguistic dim'
    elif key == 'time':
        return 'relativity'
    elif key == 'motion':
        return 'relativity'
    elif key == 'ipron':
        return 'std linguistic dim'
    elif key == 'negemo':
        return 'affective processes'
    elif key == 'work':
        return 'personal concerns'
    elif key == 'insight':
        return 'cognitive processes'
    elif key == 'posemo':
        return 'affective processes'
    elif key == 'adverb':
        return 'std linguistic dim'
    elif key == 'bio':
        return 'biological processes'
    elif key == 'percept':
        return 'perceptual processes'
    elif key == 'leisure':
        return 'personal concerns'
    elif key == 'family':
        return 'social processes'
    elif key == 'anger':
        return 'affective processes'
    elif key == 'cause':
        return 'cognitive processes'
    elif key == 'they':
        return 'std linguistic dim'
    elif key == 'quant':
        return 'std linguistic dim'
    elif key == 'home':
        return 'personal concerns'
    elif key == 'health':
        return 'personal concerns'
    elif key == 'money':
        return 'personal concerns'
    elif key == 'see':
        return 'perceptual processes'
    elif key == 'death':
        return 'personal concerns'
    elif key == 'tentat':
        return 'cognitive processes'
    elif key == 'number':
        return 'std linguistic dim'
    elif key == 'certain':
        return 'cognitive processes'
    elif key == 'discrep':
        return 'cognitive processes'
    elif key == 'hear':
        return 'perceptual processes'
    elif key == 'negate':
        return 'std linguistic dim'
    elif key == 'sad':
        return 'affective processes'
    elif key == 'friend':
        return 'social processes'
    elif key == 'body':
        return 'biological processes'
    elif key == 'anx':
        return 'affective processes'
    elif key == 'ingest':
        return 'biological processes'
    elif key == 'feel':
        return 'perceptual processes'
    elif key == 'relig':
        return 'personal concerns'
    elif key == 'sexual':
        return 'biological processes'
    elif key == 'assent':
        return 'spoken categories'
    elif key == 'swear':
        return 'std linguistic dim'
    elif key == 'filler':
        return 'spoken categories'
    elif key == 'achiev':
        return 'personal concerns' 
    elif key == 'focuspast':
        return 'std linguistic dim'
    elif key == 'focuspresent':
        return 'std linguistic dim'
    elif key == 'focusfuture':
        return 'std linguistic dim'

def tokenize(text):
    # you may want to use a smarter tokenizer
    for match in re.finditer(r'\w+', text, re.UNICODE):
        yield match.group(0)

# Define a function to calculate the Gini coefficient
def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    array = np.array(array, dtype=np.float64)
    if np.amin(array) < 0:
        array -= np.amin(array)  # Values cannot be negative
    array += 0.0000001  # Values cannot be 0
    array = np.sort(array)  # Values must be sorted
    index = np.arange(1, array.shape[0] + 1)  # Index per array element
    n = array.shape[0]
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))  # Gini coefficient


def main():
    # columns = ['social', 'relativ', 'verb', 'pronoun', 'article', 'ppron', 'space', 'conj', 'shehe', 'affect', 'auxverb', 'time', 'motion', 'ipron', 'negemo', 'work', 'insight', 'posemo', 'adverb', 'bio', 'percept', 'leisure', 'family', 'anger', 'cause', 'they', 'quant', 'home', 'health', 'money', 'see', 'death', 'tentat', 'number', 'certain', 'discrep', 'hear', 'negate', 'sad', 'friend', 'body', 'anx', 'ingest', 'feel', 'relig', 'sexual', 'assent', 'swear', 'filler', 'achiev', 'focuspast', 'focuspresent', 'focusfuture']
    columns = ['social', 'relativ', 'verb', 'pronoun', 'article', 'ppron', 'space', 'conj', 'shehe', 'affect', 'auxverb', 'time', 'motion', 'ipron', 'negemo', 'work', 'insight', 'posemo', 'adverb', 'bio', 'percept', 'leisure', 'family', 'anger', 'cause', 'they', 'quant', 'home', 'health', 'money', 'see', 'death', 'tentat', 'number', 'certain', 'discrep', 'hear', 'negate', 'sad', 'friend', 'body', 'anx', 'ingest', 'feel', 'relig', 'sexual', 'assent', 'swear']
    index_g = ['std linguistic dim', 'social processes', 'affective processes', 'cognitive processes', 'perceptual processes', 'biological processes', 'relativity', 'personal concerns', 'spoken categories']

    parse_en, category_names = liwc.load_token_parser('./DICIONÁRIOS LIWC/LIWC2015_English_Flat.dic')
    parse_pt, category_names = liwc.load_token_parser('./DICIONÁRIOS LIWC/Brazilian_Portuguese_LIWC2007_Dictionary.dic')

    path = './data/plots_en'
    text = ""
    list_files_en = os.listdir(path)
    index = [s.split(".txt")[0] for s in list_files_en]
    df_en = pd.DataFrame(columns=columns, index=index)

    for file in list_files_en:
        name_film = file.split(".txt")
        if name_film[0] not in plots["TITLE_EN"].values:
            continue

        if '.txt' not in file:
            continue

        article = open(path + "/" + file, 'r')
        content = article.read()
        
        if content != "":
            text = content

        text = re.sub(r'[^\w\s]','',text)

        text_tokens = tokenize(text)

        # now flatmap over all the categories in all of the tokens using a generator:
        text_counts = Counter(category for token in text_tokens for category in parse_en(token))
        
        for key, value in text_counts.items():
            if key in columns:
                new_key = set_keys(key)
                name = file.split(".txt")[0]
                df_en.at[name, key] = value

    df_en = df_en.fillna(0)
    for i, j in df_en.iterrows():
        df_en.loc[i] /= (df_en.loc[i].sum(axis=0))
    
    print(df_en.index)

    df = pd.read_csv("./data/oscar_full.csv", on_bad_lines='skip', sep=';')
    df_relation_names = df[["TITLE_EN", "TITLE_PT"]]
    df_relation_names = df.set_index('TITLE_PT')

    path = './data/plots_pt'
    text = ""
    list_files = os.listdir(path)
    index = [s.split(".txt")[0] for s in list_files_en]
    df_pt = pd.DataFrame(columns=columns, index=index)

    for file in list_files:
        name_film = file.split(".txt")
        if name_film[0] not in plots["TITLE_PT"].values:
            continue

        if '.txt' not in file:
            continue

        article = open(path + "/" + file, 'r')
        content = article.read()
        
        if content != "":
            text = content

        text = re.sub(r'[^\w\s]','',text)

        text_tokens = tokenize(text)

        # now flatmap over all the categories in all of the tokens using a generator:
        text_counts = Counter(category for token in text_tokens for category in parse_pt(token))
        
        for key, value in text_counts.items():
            if key in columns:
                new_key = set_keys(key)
                name_pt = file.split(".txt")[0]
                if name_pt == "Frost_Nixon":
                    name_pt = "Frost/Nixon"
                name_df = df_relation_names.at[name_pt, "TITLE_EN"]
                df_pt.at[name_df, key] = value

    df_pt = df_pt.fillna(0)
    for i, j in df_pt.iterrows():
        df_pt.loc[i] /= (df_pt.loc[i].sum(axis=0))

    print(df_en)
    print(df_pt)
    df = df_en - df_pt
    df = df.fillna(0)
    print(df)
    data = []
    for column in df:
        data.append(df[column].values)

    # sns.boxplot(y=df.columns, x=data) 
    fig = plt.figure(figsize =(10, 6))
    ax = fig.add_axes([0.06, 0.04, 0.9, 0.9])
    plt.boxplot(df, notch=False, vert=False)
    ax.set_yticklabels(columns)
    plt.title("Variação da diferença da porcentagem de ocorrência de cada grupo de categoria por filme")
    # plt.yticks("Categorias do LIWC")
    # plt.xticks("Porcentagem das palavras categorizadas como tal")
    #plt.xlim(-0.01, 0.20)
    plt.show()


if __name__ == "__main__":
    main()