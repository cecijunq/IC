import urllib.request
import wikipedia as wp
from bs4 import BeautifulSoup
from urllib.request import urlopen
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd


n_indegree = []
n_outdegree = []
def top_five_page_rank(ranks):
    top = list(reversed(sorted((rank, node) for node, rank in ranks.items()))) [:5]
    return [node for rank, node in top]

wp.set_lang("en")
filmes_indicados = {} # chave: nome do filme, valor: id (que é o valor de 'i')
grafo = nx.DiGraph()

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
    # esse arquivo contém os nomes dos filmes e os números de links, references e o tamanho do content de cada filme
    with open("./CSV METADADOS/metadados_en.csv", "r") as file:
        next(file) # pula a linha que contém o cabeçalho do arquivo .csv
        for line in file: # percorre cada linha do arquivo
            vector_data_csv = line.split(';')
            nome_filme = vector_data_csv[0]

            filme = wp.page(title=nome_filme, auto_suggest=False) # busca a página da wikipedia, cujo título é igual ao título do filme lido naquela linha do .csv
            
            # ESSA PARTE DO CÓDIGO REALIZA O PARSER DE CADA ARTIGO ARMAZENADO NA PASTA "ARTIGOS TXT"
            fp = urllib.request.urlopen(filme.url)
            html_doc = fp.read()
            soup = BeautifulSoup(html_doc, 'html.parser')
            str_article = soup.prettify()

            arr_article = []

            # esse bloco do código seleciona apenas a parte do artigo que é até antes de conter apenas links
            if '<span class="mw-headline" id="See_also">' in str_article:
                arr_article = str_article.split('<span class="mw-headline" id="See_also">')
            elif '<span class="mw-headline" id="Notes">' in str_article:
                arr_article = str_article.split('<span class="mw-headline" id="Notes">')
            elif '<span class="mw-headline" id="References">' in str_article:
                arr_article = str_article.split('<span class="mw-headline" id="References">')
            elif '<span class="mw-headline" id="Further_reading">' in str_article:
                arr_article = str_article.split('<span class="mw-headline" id="Further_reading">')
            elif '<span class="mw-headline" id="External_links">' in str_article:
                arr_article = str_article.split('<span class="mw-headline" id="External_links">')

            content_str = arr_article[0]
            content_str += "<\h2></div></div></div></main></div></div></div></body></html>" # adiciona as tags que foram "perdidas" ao chamar o método 'split'

            article_encoded = content_str.encode(encoding = 'UTF-8')

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
                    
                    print(url)
                    html_page = urlopen(url)
                    soup = BeautifulSoup(html_page, 'html.parser')
                    # obtém o nome de cada página que é referenciada no artigo em questão
                    title = soup.title.string
                    title = title.split(" - Wikipedia")
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

            # FIM DO PARSER
            

    plt.figure(1, figsize=(12,8))
    plt.axis('off')
    nx.draw_networkx(grafo, with_labels= True, node_size=200, pos=nx.spring_layout(grafo))

    grafo_ordenado = sorted(nx.strongly_connected_components(grafo), key=len, reverse=True)
    giant_component = grafo.subgraph(grafo_ordenado[0])
    #plt.show()
    plt.savefig('grafo_en.png')
    max_out = 0
    max_in = 0
    min_out = 600
    min_in = 600

    md_out = 0
    md_in = 0

    i = 0
    for tuple in grafo.out_degree():
        if i == 0:
            min_out = tuple[1]
        else:
            if tuple[1] < min_out:
                min_out = tuple[1]

        n_outdegree.append(tuple[1])
        md_out += tuple[1]
        i+=1
        if tuple[1] > max_out:
            max_out = tuple[1]

    md_out = md_out / i
    
    i = 0    
    for tuple in grafo.in_degree():
        if i == 0:
            min_in = tuple[1]
        else:
            if tuple[1] < min_in:
                min_in = tuple[1]
                
        n_indegree.append(tuple[1])
        md_in += tuple[1]
        i+=1
        if tuple[1] > max_in:
            max_in = tuple[1]

    md_in = md_in / i
    #print(n_indegree)
    #print(n_outdegree)
    #print(md_in)
    #print(md_out)
    #print(max_out)
    #print(min_out)
    #print(max_in)
    #print(min_in)

    with open("./métricas_grafo.csv", "a") as to_write:

        diametro = nx.diameter(giant_component)

        soma_in_degree = 0
        for tuple in grafo.in_degree():
            soma_in_degree += tuple[1]

        #calcula o grau médio de saída do grafo
        soma_out_degree = 0
        for tuple in grafo.out_degree():
            soma_out_degree += tuple[1]

        media_in_degree = soma_in_degree/grafo.number_of_nodes()
        media_out_degree = soma_out_degree/grafo.number_of_nodes()
        print(media_in_degree)
        print(media_out_degree)

        # calcula o betweenness médio
        soma = 0.0
        bet = nx.betweenness_centrality(grafo, normalized=False)
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

        print(nx.pagerank(grafo))

        to_write.write("Inglês;442;16845;{};{};{};{};{};{};{};{};{};{};{};{};{};{}\n".format(diametro, max_in, min_in, max_out, min_out, md_in, md_out, media_in_degree, media_out_degree, betw_md, clo_md, densidade, transitividade, pg_rnk))

        nx.write_gexf(grafo, "grafo_en.gexf")
        print(grafo)

def main():
    get_indicados()
    get_vertices()

if __name__ == "__main__":
    main()