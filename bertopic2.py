from sentence_transformers import SentenceTransformer
import wikipedia
import os
import umap
import hdbscan
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def c_tf_idf(documents, m, ngram_range=(1, 1)):
    count = CountVectorizer(ngram_range=ngram_range).fit(documents)
    t = count.transform(documents).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w)
    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)

    return tf_idf, count

def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=8):
    words = count.get_feature_names_out()
    labels = list(docs_per_topic.Topic)
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -n:]
    top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)}

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
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    # essa lista armazenará as listas dos parágrafos de cada um dos artigos
    sentences_articles = []

    # lê cada um dos arquivos .txt do seguinte diretório
    path = "./ARTIGOS TXT/artigos portugues"

    # lista o nome de todos os arquivos do diretório passado como parâmetro à função chamada
    list_files = os.listdir(path)

    # obtém o conteúdo de todos os arquivos .txt e forma uma lista composta por todos os parágrafos do texto
    for file in list_files:
        article = open(path + "/" + file, 'r')
        content = article.read()

        if "== Ver também ==" in content:
            article_as_vector_of_sentences = content.split("== Ver também ==")
        elif "== Referências ==" in content:
            article_as_vector_of_sentences = content.split("== Referências ==")
        elif "== Ligações externas ==" in content:
            article_as_vector_of_sentences = content.split("== Ligações externas ==")
        
        sentences_articles.append(article_as_vector_of_sentences[0])
        
        article_as_vector_of_sentences = content.split("\n")

        # adiciona essa lista como um elemento da lista 'sentences_article'
        for element in article_as_vector_of_sentences:
            #sentences_articles.append(element)
            
            if element == "== Ver também ==" or element == "== Referências ==":
                break
            if element != "\n" or "== " not in element or "https://" not in element:
                sentences_articles.append(element)

    print(len(sentences_articles))
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
    

    docs_df = pd.DataFrame(sentences_articles, columns=["Doc"])
    docs_df['Topic'] = cluster.labels_
    docs_df['Doc_ID'] = range(len(docs_df))
    docs_per_topic = docs_df.groupby(['Topic'], as_index = False).agg({'Doc': ' '.join})

    tf_idf, count = c_tf_idf(docs_per_topic.Doc.values, m=len(sentences_articles))

    print(type(tf_idf))
    top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=8)
    topic_sizes = extract_topic_sizes(docs_df) 
    #print(topic_sizes.head(10))
    print(type(tf_idf.T))

    #print(top_n_words)
    #print(topic_sizes)

    with open("topics_pt_t", "w") as write_topics:
        write_topics.write(str(top_n_words))
        write_topics.write("\n")
        pd.set_option('display.max_rows', None)
        write_topics.write(str(topic_sizes))

    plt.show()


if __name__ == "__main__":
    main()
