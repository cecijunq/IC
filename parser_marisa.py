import os
import wikipedia as wp
from bs4 import BeautifulSoup
from urllib.request import urlopen
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import codecs 


#Retira o elemento da pagina html
def remove_element(soup,element_name):
    for div in soup.findAll('div',{'class':element_name}):
        div.decompose()

    return soup


n_indegree = [] # armazena o grau de saída de cada vértice 
n_outdegree = [] # armazena o grau de entrada de cada vértice

def top_five_page_rank(ranks):
    top = list(reversed(sorted((rank, node) for node, rank in ranks.items()))) [:5]
    return [node for rank, node in top]

wp.set_lang("pt")
filmes_indicados = {} # chave: nome do filme, valor: id (que é o valor de 'i')
grafo = nx.DiGraph() # variável que guarda o grafo

# ARMAZENA EM UM DICIONÁRIO TODOS OS FILMES QUE TÊM PÁGINA NA WIKIPEDIA NO IDIOMA
def get_indicados():
    with open("./CSV METADADOS/metadados_pt.csv", "r") as file_get_nomes_filmes: 
        next(file_get_nomes_filmes) # pula a linha que contém o cabeçalho do arquivo .csv
        i = 0 # atua como o id do filme na lista
        for line in file_get_nomes_filmes:
            vector_data_csv = line.split(';')
            nome = vector_data_csv[0] # seleciona o nome do filme
            filmes_indicados[nome] = i
            with open("./LEGENDAS GRAFOS/legenda grafo pt.txt", "a") as file_leg:
                file_leg.write(str(i) + " " + nome + "\n")
            i+=1
        
    
def get_vertices():
    # lista todos os arquivos armazenados no diretório 'ARTIGOS HTML/inglês'
    path = "./ARTIGOS HTML/português"
    list_files = os.listdir(path) # lista o nome de todos os documentos do diretório "./ARTIGOS HTML/inglês"
    
    # acessa cada arquivo .html para lê-lo e realizar o parser
    for file in list_files:
        nome_filme = file.replace(".html", "")
        #print(nome_filme)
        film = path + '/' + file
    
        with open(film, 'r') as fp:
            content = fp.read()

            #arr_article = []

            soup = BeautifulSoup(content, 'html.parser')

            #Remove div do final
            soup = remove_element(soup,'navbox')
            soup = remove_element(soup,'navbox-styles')
            soup = remove_element(soup,'navbox authority-control')
            soup = remove_element(soup,'catlinks')

            links = list()

            for link in soup.findAll("a",href=True):
                if '/wiki/' in link['href']:
                    referenciado = link['href'].split('/wiki/')[1].replace('_', ' ')
                    if referenciado in filmes_indicados.keys() and referenciado != nome_filme and referenciado not in links:
                        links.append(referenciado)
                        print(nome_filme, referenciado)
                        grafo.add_edge(filmes_indicados[nome_filme], filmes_indicados[referenciado])

            #links = list(dict.fromkeys(links))
        
    nx.write_gexf(grafo, "grafo_pt.gexf")
    plt.show()

def main():
    get_indicados()
    get_vertices()

if __name__ == "__main__":
    main()