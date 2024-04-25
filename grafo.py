import os
import wikipedia as wp
from bs4 import BeautifulSoup
from urllib.request import urlopen
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

import community

import igraph as ig
import leidenalg as la

to_remove_en = ["Alibi (1929 film)",
"Here Comes the Navy",
"The House of Rothschild",
"The White Parade",
"Ruggles of Red Gap",
"San Francisco (1936 film)",
"A Tale of Two Cities (1935 film)",
"Four Daughters",
"Smilin' Through (1932 film)",
"Three Smart Girls",
"One Foot in Heaven",
"The Pied Piper (1942 film)",
"Wake Island (film)",
"The Human Comedy (film)",
"In Which We Serve"]


to_remove_es = ["Alibi (1929)",
"Here Comes the Navy",
"The House of Rothschild",
"The White Parade",
"Ruggles of Red Gap",
"San Francisco (filme)",
"A Tale of Two Cities (1935)",
"Four Daughters",
"Smilin' Through",
"Three Smart Girls",
"One Foot in Heaven",
"The Pied Piper",
"Wake Island",
"A Comédia Humana (1943)",
"In Which We Serve"]

def louvain_community_detector(G):
    # Run the Louvain algorithm
    partition = community.best_partition(G)
    # Draw the network with node colors based on community
    pos = nx.spring_layout(G, seed=30)
    plt.figure(figsize=(15, 7))
    nx.draw_networkx_nodes(G, pos, node_color=list(partition.values()), cmap=plt.cm.Set1, node_size=50)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.title("Detecção de Comunidades - Artigos em inglês da Wikipedia dos filmes indicados a 'Melhor Filme' no Oscar (Louvain)")
    # plt.axis("off")
    legenda = []
    # for node, community_id in partition.items():
    #     print(f"Node {node}: Community {community_id}")
    #     legenda.append(community_id)
    # plt.legend(legenda)
    nx.write_gexf(G, "comunidades_en.gexf")
    # plt.show()
    # Print the nodes and their assigned communities

#Retira o elemento da pagina html
def remove_element(soup,element_name):
    for div in soup.findAll('div',{'class':element_name}):
        div.decompose()

    return soup


n_indegree = [] # armazena o grau de saída de cada vértice 
n_outdegree = [] # armazena o grau de entrada de cada vértice

def top_five_page_rank(ranks):
    top = list(reversed(sorted((rank, node) for node, rank in ranks.items()))) [:10]
    return [node for rank, node in top]

wp.set_lang("es")
# filmes_indicados = {} # chave: nome do filme, valor: id (que é o valor de 'i')
grafo = nx.DiGraph() # variável que guarda o grafo

# ARMAZENA EM UM DICIONÁRIO TODOS OS FILMES QUE TÊM PÁGINA NA WIKIPEDIA NO IDIOMA
# def get_indicados():
#     with open("./CSV METADADOS/metadados_es.csv", "r") as file_get_nomes_filmes: 
#         next(file_get_nomes_filmes) # pula a linha que contém o cabeçalho do arquivo .csv
#         i = 0 # atua como o id do filme na lista
#         for line in file_get_nomes_filmes:
#             vector_data_csv = line.split(';')
#             nome = vector_data_csv[0] # seleciona o nome do filme
#             if nome not in to_remove_es:
#                 filmes_indicados[nome] = i
#                 # with open("./GRAFOS/LEGENDAS GRAFOS/legenda grafo en.txt", "a") as file_leg:
#                 #     file_leg.write(str(i) + " " + nome + "\n")
#                 i+=1

vector_nomes = []
def get_indicados():
    with open("./NOMES FILMES/nome filmes inglês.txt") as file:
        for line in file:
            vector_nomes.append(line.split("\n")[0])
        
    
def get_vertices():
    # lista todos os arquivos armazenados no diretório 'ARTIGOS HTML/inglês'
    path = "./ARTIGOS HTML/inglês"
    list_files = os.listdir(path) # lista o nome de todos os documentos do diretório "./ARTIGOS HTML/inglês"
    
    # acessa cada arquivo .html para lê-lo e realizar o parser
    for file in list_files:
        nome_filme = file.replace(".html", "")
        film = path + '/' + file
    
        with open(film, 'r') as fp:
            content = fp.read()

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
                    if referenciado in vector_nomes and referenciado != nome_filme:
                        links.append(referenciado)
                        #print(nome_filme, referenciado)
                        grafo.add_edge(nome_filme, referenciado)
        
    n_nos = grafo.number_of_nodes()
    n_arestas = grafo.number_of_edges()

    i = 0
    soma_out = 0
    soma_in = 0
    max_out = 0
    max_in = 0
    min_out = 600
    min_in = 600

    print(grafo.in_degree())
    print(grafo.out_degree())
    print([elem_in[1] for elem_in in grafo.in_degree])
    print([elem_out[1] for elem_out in grafo.out_degree])

    filme_max = ""
    for tuple in grafo.out_degree():
        if i == 0:
            print(f'x:{tuple[1]}')
            min_out = tuple[1]
        else:
            if tuple[1] < min_out:
                min_out = tuple[1]
        if tuple[1] != 0:
            i+=1 
        n_outdegree.append(tuple[1])
        soma_out += tuple[1]
        if tuple[1] > max_out:
            max_out = tuple[1]

    md_out = soma_out / n_nos
        
    i = 0
    for tuple in grafo.in_degree():
        if i == 0:
            print(f'y:{tuple[1]}')
            min_in = tuple[1]
        else:
            if tuple[1] < min_in:
                min_in = tuple[1]
        if tuple[1] != 0:
            i+=1 
        n_indegree.append(tuple[1])
        soma_in += tuple[1]
        if tuple[1] > max_in:
            max_in = tuple[1]

    md_in = soma_in / n_nos
    

    grafo_ordenado = sorted(nx.strongly_connected_components(grafo), key=len, reverse=True)
    giant_component = grafo.subgraph(grafo_ordenado[0])
    diametro = nx.diameter(giant_component)

    # calcula o betweenness médio
    soma = 0.0
    bet = nx.betweenness_centrality(grafo, normalized=True)
    between = pd.DataFrame.from_dict(data=bet, orient='index')

    for element in bet.values():
        soma += element
    betw_md = soma/grafo.number_of_nodes()

    soma = 0.0
    # calcula o closeness médio
    close = nx.closeness_centrality(grafo)
    closeness = pd.DataFrame.from_dict(data=close, orient='index')

    for element in close.values():
        soma += element
    clo_md = soma/grafo.number_of_nodes()

    densidade = nx.density(grafo)
    transitividade = nx.transitivity(grafo)
    pg_rnk = top_five_page_rank(nx.pagerank(grafo))

    print("Inglês;{};{};{};{};{};{};{};{};{};{};{};{};{};{}\n".format(n_nos, n_arestas, diametro, max_in, min_in, max_out, min_out, md_in, md_out, betw_md, clo_md, densidade, transitividade, pg_rnk))
    with open("./métricas_grafo.csv", "a") as to_write:
        to_write.write("Inglês;{};{};{};{};{};{};{};{};{};{};{};{};{}\n".format(n_nos, n_arestas, diametro, max_in, min_in, max_out, min_out, md_in, betw_md, clo_md, densidade, transitividade, pg_rnk))
    nx.write_gexf(grafo, "grafo_en2.gexf")

    plt.show()

    louvain_community_detector(grafo.to_undirected())
    # partition = la.find_partition(grafo, la.ModularityVertexPartition)
    # ig.plot(partition)
    

def main():
    get_indicados()
    get_vertices()

if __name__ == "__main__":
    main()