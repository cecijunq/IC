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
    index = ['function', 'prep', 'social', 'relativ', 'verb', 'pronoun', 'article', 'drives', 'focuspresent', 'ppron', 'cogproc', 'space', 'conj', 'shehe', 'affect', 'male', 'auxverb', 'time', 'power', 'adj', 'motion', 'ipron', 'negemo', 'work', 'female', 'affiliation', 'insight', 'posemo', 'adverb', 'bio', 'percept', 'focuspast', 'differ', 'compare', 'achiev', 'leisure', 'family', 'interrog', 'anger', 'cause', 'they', 'focusfuture', 'reward', 'quant', 'home', 'health', 'money', 'see', 'death', 'risk', 'tentat', 'number', 'certain', 'discrep', 'hear', 'negate', 'sad', 'friend', 'body', 'anx', 'ingest', 'feel', 'relig', 'sexual', 'informal', 'nonflu', 'we', 'you', 'assent', 'i', 'swear', 'filler', 'netspeak', 'cogmech', 'incl', 'excl', 'humans', 'inhib']
    data_en = [0 for i in range(len(index))]
    data_pt = [0 for i in range(len(index))]
    data = {"Inglês": data_en,
            "Português": data_pt}
    
    df = pd.DataFrame(data=data, index=index) 

    # liwc = LIWC("Brazilian_Portuguese_LIWC2007_Dictionary.dic")
    parse_en, category_names = liwc.load_token_parser('./DICIONÁRIOS LIWC/LIWC2015_English_Flat.dic')
    parse_pt, category_names = liwc.load_token_parser('./DICIONÁRIOS LIWC/Brazilian_Portuguese_LIWC2007_Dictionary.dic')

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
    # now flatmap over all the categories in all of the tokens using a generator:
    text_counts = Counter(category for token in text_tokens for category in parse_en(token))

    for key, value in text_counts.items():
        df.at[key, "Inglês"] = value


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
        if key == 'funct':
            df.at['function', "Português"] = value

        elif key == 'preps':
            df.at['prep', "Português"] = value

        elif key == 'nonfl':
            df.at['nonflu', "Português"] = value

        elif key == 'achieve':
            df.at['achiev', "Português"] = value

        elif key == 'past':
            df.at['focuspast', "Português"] = value

        elif key == 'future':
            df.at['focusfuture', "Português"] = value

        elif key == 'present':
            df.at['focuspresent', "Português"] = value

        else:
            df.at[key, "Português"] = value

    # and print the results:
    pd.set_option('display.max_rows', df.shape[0]+1)
    print(df)

    # Normalize the data
    df_normalized = df.div(df.sum(axis=1), axis=0)
    
    # Apply the Gini coefficient calculation to each row
    gini_values = df.apply(lambda row: gini(row), axis=1)

    # Select the top 10 categories based on the Gini coefficient
    top_10_categories = gini_values.sort_values(ascending=False).head(10).index

    # Filter the DataFrame to only include the top 10 categories
    df_top_10 = df.loc[top_10_categories]

    # Normalize the filtered data
    df_top_10_normalized = df_top_10.div(df_top_10.sum(axis=1), axis=0)

    # Plot the heatmap
    plt.figure(figsize=(15, 30))  # Increase figure size to fit all categories
    sns.heatmap(df_top_10_normalized, annot=False, cmap="coolwarm")
    plt.xlabel('')
    plt.ylabel('')
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.title('Normalized Scores for English and Portuguese Wikipedia Articles')
    plt.show()


if __name__ == "__main__":
    main()