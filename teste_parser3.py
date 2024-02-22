import os
import wikipedia as wp
from bs4 import BeautifulSoup
from urllib.request import urlopen
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import codecs 


n_indegree = []
n_outdegree = []
def top_five_page_rank(ranks):
    top = list(reversed(sorted((rank, node) for node, rank in ranks.items()))) [:5]
    return [node for rank, node in top]

wp.set_lang("en")
filmes_indicados = {} # chave: nome do filme, valor: id (que é o valor de 'i')
grafo = nx.DiGraph()

# ARMAZENA EM UM DICIONÁRIO TODOS OS FILMES QUE TÊM PÁGINA NA WIKIPEDIA NO IDIOMA
def get_indicados():
    with open("./CSV METADADOS/metadados_en.csv", "r") as file_get_nomes_filmes: 
        next(file_get_nomes_filmes) # pula a linha que contém o cabeçalho do arquivo .csv
        i = 0 # atua como o id do filme na lista
        for line in file_get_nomes_filmes:
            vector_data_csv = line.split(';')
            nome = vector_data_csv[0] # seleciona o nome do filme
            filmes_indicados[nome] = i
            with open("./LEGENDAS GRAFOS/legenda grafo en.txt", "a") as file_leg:
                file_leg.write(str(i) + " " + nome + "\n")
            i+=1
        
    
def get_vertices():
    # lista todos os arquivos armazenados no diretório 'ARTIGOS HTML/inglês'
    path = "./ARTIGOS HTML/inglês"
    list_files = os.listdir(path)
    
    # acessa cada arquivo .html para lê-lo e realizar o parser
    for file in list_files:
        nome_filme = file.replace(".html", "")
        print(nome_filme)
        film = path + '/' + file
    
        with open(film, 'r') as fp:
            content = fp.read()

            arr_article = []

            # esse bloco do código seleciona apenas a parte do artigo que é até antes de conter apenas links
            if '<span class="mw-headline" id="See_also">' in content:
                arr_article = content.split('<span class="mw-headline" id="See_also">')
            elif '<span class="mw-headline" id="Notes">' in content:
                arr_article = content.split('<span class="mw-headline" id="Notes">')
            elif '<span class="mw-headline" id="References">' in content:
                arr_article = content.split('<span class="mw-headline" id="References">')
            elif '<span class="mw-headline" id="Further_reading">' in content:
                arr_article = content.split('<span class="mw-headline" id="Further_reading">')
            elif '<span class="mw-headline" id="External_links">' in content:
                arr_article = content.split('<span class="mw-headline" id="External_links">')

            content_parser = arr_article[0]
            content_parser += "<\h2></div></div></div></main></div></div></div></body></html>" # adiciona as tags que foram "perdidas" ao chamar o método 'split'

            #print(content_parser)
            article_encoded = content_parser.encode(encoding = 'UTF-8')

            soup = BeautifulSoup(article_encoded, 'html.parser')

            referenciados = {}

            # seleciona o atributo 'href' de todas as tags <a> que existem no html do artigo em questão
            for tag in soup.select('a', href=True):
                url = tag.get("href")
                if url is not None and '/' in url:

                    if "https" not in url and "//en.wikipedia.org" not in url and "/wiki/" in url:
                        #print(url)
                        url = url.replace("/wiki/", "https://en.wikipedia.org/wiki/")

                    elif "https:" not in url and "//en.wikipedia.org" in url:
                        url = "https:" + url

                    else:
                        continue
                    
                    #print(url)
                    html_page = urlopen(url)
                    soup = BeautifulSoup(html_page, 'html.parser')
                    # obtém o nome de cada página que é referenciada no artigo em questão
                    title = soup.title.string
                    title = title.split(" - Wikipedia")
                    #print(title[0])
                    #print(nome_filme, title[0])
                    
                    referenciado = str(title[0])
                    referenciado = referenciado.strip()
                    if referenciado != 'None':
                        # checa se o título da página referenciada está na lista dos filmes indicados à categoria de "Melhor Filme"
                        if referenciado in filmes_indicados.keys() and referenciado != nome_filme:
                            print(nome_filme, referenciado)
                            grafo.add_edge(filmes_indicados[nome_filme], filmes_indicados[referenciado])
                        #referenciados.append(referenciado)

            #referenciados = list(dict.fromkeys(referenciados))
            #print(referenciados)
            fp.close()

def main():
    get_vertices()

if __name__ == "__main__":
    main()