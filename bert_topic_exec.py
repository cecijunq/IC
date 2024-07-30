import pandas as pd 
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import os
import re,sys
import umap
import hdbscan
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import wikipedia

from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired
from bertopic.representation import MaximalMarginalRelevance
from sklearn.decomposition import PCA
from IPython.display import Markdown, display

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

pd.set_option('display.max_columns', None)

def extract_topic_sizes(df):
    topic_sizes = (df.groupby(['Topic'])
                     .Doc
                     .count()
                     .reset_index()
                     .rename({"Topic": "Topic", "Doc": "Size"}, axis='columns')
                     .sort_values("Size", ascending=False))
    return topic_sizes

def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=8):
    print(type(count))
    words = count.get_feature_names_out()
    labels = list(docs_per_topic.Topic)
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -n:]
    top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)}

    return top_n_words

def c_tf_idf(documents, m, ngram_range=(1, 1)):
    count = CountVectorizer(ngram_range=ngram_range, stop_words=stopwords.words('english')).fit(documents)
    t = count.transform(documents).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w)
    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)

    return tf_idf, count

def rescale(x, inplace=False):
    """ Rescale an embedding so oesimization will not have convergence issues.
    """
    if not inplace:
        x = np.array(x, copy=True)

    x /= np.std(x[:, 0]) * 10000

    return x


def func(x):
    ar = x[0].split('\n')
    return ar[0]


def func2(x):
    ar = x.split('\n')
    return ar[0]


sentences_articles = []

# path = './ARTIGOS TXT/artigos inglês'
path = './data/plots_en'
list_files = os.listdir(path)

list_files = ['Black Panther (film).txt', 'The Bridge on the River Kwai.txt', 'Philomena (film).txt', 'Mutiny on the Bounty (1962 film).txt', 'Out of Africa (film).txt', 'The Godfather Part II.txt', 'Wuthering Heights (1939 film).txt', 'Traffic (2000 film).txt', 'For Whom the Bell Tolls (film).txt', 'The Shawshank Redemption.txt', 'Casablanca (film).txt', 'All This, and Heaven Too.txt', 'The Killing Fields (film).txt', 'The Cider House Rules (film).txt', "Prizzi's Honor.txt", 'The Godfather.txt', 'A Room with a View (1985 film).txt', 'Dr. Strangelove.txt', 'Airport (1970 film).txt', 'The Snake Pit.txt', 'Capote (film).txt', 'Chariots of Fire.txt', '49th Parallel (film).txt', 'The Awful Truth.txt', 'Boyhood (2014 film).txt', 'Cimarron (1931 film).txt', 'The Lives of a Bengal Lancer (film).txt', 'The Diary of Anne Frank (1959 film).txt', 'Sergeant York (film).txt', 'American Hustle.txt', '12 Angry Men (1957 film).txt', 'Friendly Persuasion (1956 film).txt', 'Grand Hotel (1932 film).txt', 'The Insider (film).txt', 'The Philadelphia Story (film).txt', 'My Fair Lady (film).txt', 'The Big House (1930 film).txt', 'Mutiny on the Bounty (1935 film).txt', 'Citizen Kane.txt', 'Gangs of New York.txt', "The Nun's Story (film).txt", 'Shine (film).txt', 'Ivanhoe (1952 film).txt', 'Dodsworth (film).txt', 'Missing (1982 film).txt', 'Sayonara.txt', 'Four Weddings and a Funeral.txt', 'Gaslight (1944 film).txt', 'The Alamo (1960 film).txt', 'The Irishman.txt', 'The Remains of the Day (film).txt', 'Hamlet (1948 film).txt', 'Naughty Marietta (film).txt', 'Foreign Correspondent (film).txt', 'Ben-Hur (1959 film).txt', 'Captain Blood (1935 film).txt', "The King's Speech.txt", 'Decision Before Dawn.txt', 'Quo Vadis (1951 film).txt', 'The Color Purple (1985 film).txt', 'How the West Was Won (film).txt', 'The Last Emperor.txt', 'Gladiator (2000 film).txt', 'Trader Horn (1931 film).txt', 'Raiders of the Lost Ark.txt', 'West Side Story (2021 film).txt', 'Manchester by the Sea (film).txt', 'Pygmalion (1938 film).txt', 'The Deer Hunter.txt', 'The Curious Case of Benjamin Button (film).txt', 'Viva Villa!.txt', 'All Quiet on the Western Front (2022 film).txt', 'Midnight in Paris.txt', 'Green Book (film).txt', "The Razor's Edge (1946 film).txt", 'Gone with the Wind (film).txt', 'Saving Private Ryan.txt', 'The Mission (1986 film).txt', 'Arrowsmith (film).txt', 'Howards End (film).txt', 'The Robe (film).txt', 'Cabaret (1972 film).txt', 'The Private Life of Henry VIII.txt', 'A Man for All Seasons (1966 film).txt', 'Disraeli (1929 film).txt', 'The Imitation Game.txt', 'The Life of Emile Zola.txt', 'Ford v Ferrari.txt', 'Les Misérables (2012 film).txt', 'Midnight Express (film).txt', 'Shakespeare in Love.txt', 'Little Women (1933 film).txt']

articles = pd.DataFrame()
pattern = re.compile(r'== Plot|Synopsis ==\n(.*?)\n== (\S+) ==', re.DOTALL)

for file in list_files:
    if '.txt' not in file:
        continue
    # print(file)

    article = open(path + "/" + file, 'r')
    content = article.read()
    
    if content != "":
        sentences_articles.append(content)
    
    # if re.search(pattern, content):
    #     plot_text = re.search(pattern, content).group(1)
    # else: # obtain the summary if plot or synopsis is absent
    #         # print(file)
    #         # sys.exit()
    #         continue  
   

    # if "== See also ==" in content:
    #     article_as_vector_of_sentences = content.split("== See also ==")
    # elif "== References ==" in content:
    #     article_as_vector_of_sentences = content.split("== References ==")
    # elif "== External links ==" in content:
    #     article_as_vector_of_sentences = content.split("== External links ==")
        
    # sentences_articles.append(article_as_vector_of_sentences[0])


# sentence_model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v1")
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
#sentence_model = SentenceTransformer("rufimelo/Legal-BERTimbau-large-TSDAE-v4-sts")
embeddings = sentence_model.encode(sentences_articles, show_progress_bar=True)

# Initialize and rescale PCA embeddings
pca_embeddings = rescale(PCA(n_components=5).fit_transform(embeddings))

#Reduce the impact of frequent words
ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

#Remove stopwords
vectorizer_model = CountVectorizer(stop_words=stopwords.words('english'))

#KeyBERT-Inspired model to reduce the appearance of stop words
# representation_model = KeyBERTInspired()
representation_model = MaximalMarginalRelevance(diversity=0.4)


umap_model= umap.UMAP(n_neighbors=8,n_components=5,min_dist=0.0,metric='cosine', random_state=2023).fit_transform(embeddings)
# umap_model= umap.UMAP(n_neighbors=15,n_components=5,min_dist=0.0,metric='cosine',low_memory=True,init=pca_embeddings,random_state=42).fit_transform(embeddings)

hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=10,metric='euclidean', min_samples=3,prediction_data=True).fit(umap_model)

topic_model = BERTopic(embedding_model=sentence_model, language='English',
                       verbose=True,calculate_probabilities=True,
                       vectorizer_model=vectorizer_model,ctfidf_model=ctfidf_model,min_topic_size = 80,representation_model=representation_model,hdbscan_model=hdbscan_model)

topics, probs = topic_model.fit_transform(sentences_articles,embeddings)

# add
docs_df = pd.DataFrame(sentences_articles, columns=["Doc"])
docs_df['Topic'] = hdbscan_model.labels_
docs_df['Doc_ID'] = range(len(docs_df))
docs_per_topic = docs_df.groupby(['Topic'], as_index = False).agg({'Doc': ' '.join})

tf_idf, count = c_tf_idf(docs_per_topic.Doc.values, m=len(sentences_articles))

top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=10)
topic_sizes = extract_topic_sizes(docs_df) 
#fim add

# Create a dataframe with document metadata
df = topic_model.get_document_info(sentences_articles)

print(topic_model.get_topic_info().head(10).set_index('Topic')[
   ['Count', 'Name', 'Representation']])

# print(topic_model.visualize_barchart(top_n_topics = 20, n_words = 20))

with open("topics_pt_t3", "w") as write_topics:
    write_topics.write(str(top_n_words))
    write_topics.write("\n")
    pd.set_option('display.max_rows', None)
    write_topics.write(str(topic_sizes))
print(df.columns)


print(df.head())

df2 = df[["Topic", "Name", "Representation"]]
#df2 = df2.groupby("Name").count().reset_index()
df2.to_csv("topics3.csv")

df3 = df[["Topic", "Probability", "Document"]]
df3['Document'] = df3['Document'].apply(lambda x: func2(x))
df3.to_csv("probability_per_doc3.csv")

df4 = df[['Top_n_words']]
df4.drop_duplicates(inplace=True)
df4.to_csv("top_n_words3")

df["Document"] = df3['Document'].apply(lambda x: func2(x))
df["Representative_Docs"] = df['Representative_Docs'].apply(lambda x: func(x))
df.to_csv("df3_2.csv")
