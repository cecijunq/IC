from sentence_transformers import SentenceTransformer
import os
import umap
import hdbscan
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def c_tf_idf(documents, m, ngram_range=(1, 1)):
    count = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(documents)
    t = count.transform(documents).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w)
    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)

    return tf_idf, count

def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20):
    words = count.get_feature_names()
    labels = list(docs_per_topic.Topic)
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -n:]
    top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)}
    print(top_n_words)
    return top_n_words

def extract_topic_sizes(df):
    topic_sizes = (df.groupby(['Topic'])
                     .Doc
                     .count()
                     .reset_index()
                     .rename({"Topic": "Topic", "Doc": "Size"}, axis='columns')
                     .sort_values("Size", ascending=False))
    return topic_sizes

def main():
    # cria um modelo, cujo nome é 'best-picture-oscars'
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # essa lista armazenará as listas dos parágrafos de cada um dos artigos
    sentences_articles = []

    # lê cada um dos arquivos .txt do seguinte diretório
    path = "./ARTIGOS TXT/artigos inglês"

    # lista o nome de todos os arquivos do diretório passado como parâmetro à função chamada
    list_files = os.listdir(path)

    # obtém o conteúdo de todos os arquivos .txt e forma uma lista composta por todos os parágrafos do texto
    for file in list_files:
        #print(path + "/" + file)
        article = open(path + "/" + file, 'r')
        content = article.read()
        #print(type(content))

        # transforma cada artigo em um vetor formado pelos parágrafos do txt
        #article_as_vector_of_sentences = content.split("\n")

        # adiciona essa lista como um elemento da lista 'sentences_article'
        sentences_articles.append(content)
        # sentences_articles.append(article_as_vector_of_sentences) # adiciona à lista formada pelos parágrafos de todos os txt

        #with open("docs_bertopic.csv", "a") as f:
            #f.write(content)

    #print(len(sentences_articles))
    # sentenças são codificadas. Transforma os documentos em vetores de dimensão 512
    embeddings = model.encode(sentences_articles, show_progress_bar=True)

    umap_embeddings = umap.UMAP(n_neighbors=15,
                                n_components=2,
                                min_dist=0.0,
                                metric='cosine').fit_transform(embeddings)

    cluster = hdbscan.HDBSCAN(min_cluster_size=15,
                            metric='euclidean',
                            cluster_selection_method='eom').fit(umap_embeddings)

    result = pd.DataFrame(umap_embeddings, columns=['x', 'y'])
    result['labels'] = cluster.labels_

    # Visualize clusters
    fig, ax = plt.subplots(figsize=(20, 10))
    outliers = result.loc[result.labels == -1, :]
    clustered = result.loc[result.labels != -1, :]

    plt.scatter(outliers.x, outliers.y, color='#BDBDBD')
    plt.scatter(clustered.x, clustered.y, c=clustered.labels, cmap='hsv_r')
    plt.colorbar()
    plt.show()

    docs_df = pd.DataFrame(sentences_articles, columns=["Doc"])
    docs_df['Topic'] = cluster.labels_
    docs_df['Doc_ID'] = range(len(docs_df))
    docs_per_topic = docs_df.groupby(['Topic'], as_index = False).agg({'Doc': ' '.join})

    tf_idf, count = c_tf_idf(docs_per_topic.Doc.values, m=len(sentences_articles))

    top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=2)
    topic_sizes = extract_topic_sizes(docs_df); topic_sizes.head(10)

    print(top_n_words)

if __name__ == "__main__":
    main()