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

# def gini(x):
#     print(type(x))
#     print(x)
#     # (Warning: This is a concise implementation, but it is O(n**2)
#     # in time and memory, where n = len(x).  *Don't* pass in huge
#     # samples!)

#     # Mean absolute difference
#     mad = np.abs(np.subtract.outer(x, x)).mean()
#     # Relative mean absolute difference
#     rmad = mad/np.mean(x)
#     # Gini coefficient
#     g = 0.5 * rmad
#     return g


def main():
    index2 = ['social', 'relativ', 'verb', 'pronoun', 'article', 'ppron', 'space', 'conj', 'shehe', 'affect', 'auxverb', 'time', 'motion', 'ipron', 'negemo', 'work', 'insight', 'posemo', 'adverb', 'bio', 'percept', 'leisure', 'family', 'anger', 'cause', 'they', 'quant', 'home', 'health', 'money', 'see', 'death', 'tentat', 'number', 'certain', 'discrep', 'hear', 'negate', 'sad', 'friend', 'body', 'anx', 'ingest', 'feel', 'relig', 'sexual', 'assent', 'swear', 'filler', 'achiev', 'focuspast', 'focuspresent', 'focusfuture']
    index = ['std linguistic dim', 'social processes', 'affective processes', 'cognitive processes', 'perceptual processes', 'biological processes', 'relativity', 'personal concerns', 'spoken categories']
    data_en = [0 for i in range(len(index))]
    data_pt = [0 for i in range(len(index))]
    data = {"Inglês": data_en,
            "Português": data_pt}
    
    df = pd.DataFrame(data=data, index=index)

    # liwc = LIWC("Brazilian_Portuguese_LIWC2007_Dictionary.dic")
    parse_en, category_names = liwc.load_token_parser('./DICIONÁRIOS LIWC/LIWC2015_English_Flat.dic')
    parse_pt, category_names = liwc.load_token_parser('./DICIONÁRIOS LIWC/Brazilian_Portuguese_LIWC2007_Dictionary.dic')

    # file = './data/plots_en/Zorba the Greek (film).txt'
    # article = open(file, 'r')
    # content = article.read()
    path = './data/plots_en'
    text = ""
    list_files = os.listdir(path)

    for file in list_files:
        if '.txt' not in file:
            continue

        article = open(path + "/" + file, 'r')
        content = article.read()
        
        if content != "":
            text += content

    text = re.sub(r'[^\w\s]','',text)

    text_tokens = tokenize(text)

    categorias = ['Palavra', 'social', 'relativ', 'verb', 'pronoun', 'article', 'ppron', 'space', 'conj', 'shehe', 'affect', 'auxverb', 'time', 'motion', 'ipron', 'negemo', 'work', 'insight', 'posemo', 'adverb', 'bio', 'percept', 'leisure', 'family', 'anger', 'cause', 'they', 'quant', 'home', 'health', 'money', 'see', 'death', 'tentat', 'number', 'certain', 'discrep', 'hear', 'negate', 'sad', 'friend', 'body', 'anx', 'ingest', 'feel', 'relig', 'sexual', 'assent', 'swear', 'filler', 'achiev', 'focuspast', 'focuspresent', 'focusfuture']
    df_palavras = pd.DataFrame(columns=categorias)
    df_palavras['Palavra'] = pd.Series(dtype='str')
    series_cat = pd.Series(categorias)
    # print(df_palavras)
    with open("./palavras_categoria.csv", 'a') as file:
        for token in list(text_tokens):
            for category in parse_en(token):
                file.write(f'{token};{category}\n')
    """for token in list(text_tokens):
        # if token not in df_palavras['Palavra'].tolist():
        #     print(token)
        #     df_palavras.loc[len(df_palavras.index)] = [token] + [0 for i in range(len(categorias)-1)]
        for category in parse_en(token):
            if category in categorias:
                if token in df_palavras['Palavra'].values:
                    # Update existing row
                    row_index = df_palavras.index[df_palavras['Palavra'] == token].tolist()[0]
                    df_palavras.at[row_index, category] = 1
            else:
                # Add new row
                new_row = {cat: 0 for cat in categorias}
                new_row['Palavra'] = token
                new_row[category] = 1
                df_palavras = df_palavras.append(new_row, ignore_index=True)

    # Fill NaN values with 0
    df_palavras = df_palavras.fillna(0)
    print(df_palavras)"""
            #file.write(f'{token};{category}\n')
    #df_palavras.set_index(['Palavra'], inplace=True) 
                # print(f'aqui: {token}, {category}')
                # data = pd.DataFrame({"Palavra": token, "Categoria": category})
                # df_palavras = df_palavras.append({"Palavra": token, "Categoria": category}, ignore_index=True)
                # df_palavras = pd.concat([df_palavras, data], axis=0)
                # df_palavras.loc[len(df_palavras.index)] = [token, category] 

    
    # now flatmap over all the categories in all of the tokens using a generator:
    text_counts = Counter(category for token in text_tokens for category in parse_en(token))
    #print(text_counts)
    
    for key, value in text_counts.items():
        if key in index2:
            new_key = set_keys(key)
            df.at[new_key, "Inglês"] += value

    #print(len(text_counts))

    path = './data/plots_pt'
    text = ""
    list_files = os.listdir(path)

    for file in list_files:
        if '.txt' not in file:
            continue

        article = open(path + "/" + file, 'r')
        content = article.read()
        
        if content != "":
            text += content

    text = re.sub(r'[^\w\s]','',text)

    text_tokens = tokenize(text)
    # now flatmap over all the categories in all of the tokens using a generator:

    text_counts = Counter(category for token in text_tokens for category in parse_pt(token))

    for key, value in text_counts.items():
        #print(key, value)
        if key in index2:
            if key == 'achieve':
                new_key = set_keys('achiev')
                #df.at['achiev', "Português"] = value

            elif key == 'past':
                new_key = set_keys('focuspast')
                #df.at['focuspast', "Português"] = value

            elif key == 'future':
                new_key = set_keys('focusfuture')
                #df.at['focusfuture', "Português"] = value

            elif key == 'present':
                new_key = set_keys('focuspresent')
                #df.at['focuspresent', "Português"] = value

            new_key = set_keys(key)
            df.at[new_key, "Português"] += value

    # and print the results:
    pd.set_option('display.max_rows', df.shape[0]+1)
    df.to_csv("resultado_liwc.csv")

    # Normalize the data
    ser = df.sum(axis=0)
    df['Inglês'] = df['Inglês'] / ser[0]
    df['Português'] = df['Português'] / ser[1]
    #print(df.sum(axis=0))

    df_normalized = df.div(df.sum(axis=1), axis=0)
    #print(df_normalized.sum(axis=1))
    
    # Apply the Gini coefficient calculation to each row
    gini_values = df_normalized.apply(lambda row: gini(row), axis=1)
    #gini_values = df.apply(lambda s: gini(s), axis=1).sort_values(ascending=False).head(10).index
    #print(gini_values)
    #gini_values = df.apply(lambda column: gini(column))

    #print(gini_values)
    # Select the top 10 categories based on the Gini coefficient
    top_10_categories = gini_values.sort_values(ascending=False).index

    # Filter the DataFrame to only include the top 10 categories
    df_top_10 = df_normalized.loc[top_10_categories]

    # Normalize the filtered data
    df_top_10_normalized = df_top_10.div(df_top_10.sum(axis=1), axis=0)
    # print(df_top_10.sum(axis=1))
    # print(df_top_10.sum(axis=0))
    # print(df_top_10_normalized.sum(axis=1))
    # print(df_top_10_normalized.sum(axis=0))

    # Plot the heatmap
    plt.figure(figsize=(15, 12))  # Increase figure size to fit all categories
    sns.heatmap(df, annot=False, cmap="coolwarm")
    # sns.heatmap(df_normalized, annot=False, cmap="coolwarm")
    plt.xlabel('')
    plt.ylabel('')
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.title('Normalized Scores for English and Portuguese Wikipedia Articles')
    plt.show()


if __name__ == "__main__":
    main()