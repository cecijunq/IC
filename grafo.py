import wikipedia as wp
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

n_indegree = []
n_outdegree = []
def top_five_page_rank(ranks):
    top = list(reversed(sorted((rank, node) for node, rank in ranks.items()))) [:5]
    return [node for rank, node in top]

wp.set_lang("es")
filmes_indicados = {} # chave: nome do filme, valor: id (que é o valor de 'i')
grafo = nx.DiGraph()

def get_indicados():
    with open("./CSV METADADOS/metadados_es.csv", "r") as file_get_nomes_filmes: 
        next(file_get_nomes_filmes) # pula a linha que contém o cabeçalho do arquivo .csv
        i = 0 # atua como o id do filme na lista
        for line in file_get_nomes_filmes:
            vector_data_csv = line.split(';')
            nome = vector_data_csv[0] # seleciona o nome do filme
            filmes_indicados[nome] = i
            with open("./legenda grafo.txt", "a") as file_leg:
                file_leg.write(str(i) + " " + nome + "\n")
            i+=1


def get_vertices():
    # esse arquivo contém os nomes dos filmes e os números de links, references e o tamanho do content de cada filme
    with open("./CSV METADADOS/metadados_es.csv", "r") as file:
        next(file) # pula a linha que contém o cabeçalho do arquivo .csv
        for line in file: # percorre cada linha do arquivo
            vector_data_csv = line.split(';')
            nome_filme = vector_data_csv[0]

            links_filme = wp.page(title=nome_filme, auto_suggest=False) # busca a página da wikipedia, cujo título é igual ao título do filme lido naquela linha do .csv
            lista_references = list(links_filme.links) # armazena em uma lista o nome de todas as páginas que a página atual referencia

            # percorre cada elemento da lista que armazena os "references" do filme de "line"
            for link in lista_references:
                if link in filmes_indicados.keys():
                    print(nome_filme, link)
                    grafo.add_edge(filmes_indicados[nome_filme], filmes_indicados[link])

    plt.figure(1, figsize=(12,8))
    plt.axis('off')
    nx.draw_networkx(grafo, with_labels= True, node_size=200, pos=nx.spring_layout(grafo))

    #grafo_ordenado = sorted(nx.strongly_connected_components(grafo), key=len, reverse=True)
    #giant_component = grafo.subgraph(grafo_ordenado[0])
    #plt.show()
    plt.savefig('grafo_es.png')
    max_out = 0
    max_in = 0
    min_out = 600
    min_in = 600

    md_out = 0
    md_in = 0
    i = 0

    for tuple in grafo.out_degree():
        n_outdegree.append(tuple[1])
        md_out += tuple[1]
        i+=1
        if tuple[1] > max_out:
            max_out = tuple[1]
        if tuple[1] < min_out:
            min_out = tuple[1]

    md_out = md_out / i
    i = 0
        
    for tuple in grafo.in_degree():
        n_indegree.append(tuple[1])
        md_in += tuple[1]
        i+=1
        if tuple[1] > max_out:
            max_in = tuple[1]
        if tuple[1] < min_out:
            min_in = tuple[1]

    md_out = md_out / i
    #print(n_indegree)
    #print(n_outdegree)
    #print(md_in)
    #print(md_out)
    #print(max_out)
    #print(min_out)
    #print(max_in)
    #print(min_in)

    #with open("./métricas_grafo.csv", "a") as to_write:

        #diametro = nx.diameter(giant_component)

        #soma_in_degree = 0
        #for tuple in grafo.in_degree():
        #    soma_in_degree += tuple[1]

        ## calcula o grau médio de saída do grafo
        #soma_out_degree = 0
        #for tuple in grafo.out_degree():
        #    soma_out_degree += tuple[1]

        #media_in_degree = soma_in_degree/grafo.number_of_nodes()
        #media_out_degree = soma_out_degree/grafo.number_of_nodes()

            # calcula o betweenness médio
#       soma = 0.0
#       bet = nx.betweenness_centrality(grafo, normalized=False)
#       between = pd.DataFrame.from_dict(data=bet, orient='index')

#       for element in bet.values():
#           soma += element
#       betw_md = soma/grafo.number_of_nodes()

#       soma = 0.0
#       # calcula o closeness médio
#       close = nx.closeness_centrality(grafo)
#       closeness = pd.DataFrame.from_dict(data=close, orient='index')

#       for element in close.values():
#           soma += element
#       clo_md = soma/grafo.number_of_nodes()

#       densidade = nx.density(grafo)
#       transitividade = nx.transitivity(grafo)
#       pg_rnk = top_five_page_rank(nx.pagerank(grafo))

#       print(nx.pagerank(grafo))

#       to_write.write("Espanhol;442;16845;{};{};{};{};{};{};{};{}\n".format(diametro, media_in_degree, media_out_degree, betw_md, clo_md, densidade, transitividade, pg_rnk))

    nx.write_gexf(grafo, "grafo_es.gexf")
    #print(grafo)

def main():
    get_indicados()
    get_vertices()

if __name__ == "__main__":
    main()