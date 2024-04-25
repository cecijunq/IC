import wikipedia as wp
import os
from bs4 import BeautifulSoup


def remove_element(soup,element_name):
    for div in soup.findAll('div',{'class':element_name}):
        div.decompose()

    return soup

vector_nomes = []
def get_nomes():
    with open("./NOMES FILMES/nome filmes português.txt") as file:
        for line in file:
            vector_nomes.append(line.split("\n")[0])

def main():
    get_nomes()
    print(vector_nomes)
    dict_ref = {}

    path = "./ARTIGOS HTML/português"
    list_files = os.listdir(path) # lista o nome de todos os documentos do diretório "./ARTIGOS HTML/inglês"
        
    referencias_versao = []

    # acessa cada arquivo .html para lê-lo e realizar o parser
    for file in list_files:
        nome_filme = file.replace(".html", "")
        film = path + '/' + file
        
        n_referencias = 0
        with open(film, 'r') as fp:
            content = fp.read()

            soup = BeautifulSoup(content, 'html.parser')

            #Remove div do final
            soup = remove_element(soup,'navbox')
            soup = remove_element(soup,'navbox-styles')
            soup = remove_element(soup,'navbox authority-control')
            soup = remove_element(soup,'catlinks')

            for link in soup.findAll("a",href=True):
                if '/wiki/' in link['href']:
                    referenciado = link['href'].split('/wiki/')[1].replace('_', ' ')
                    if referenciado in vector_nomes and referenciado != nome_filme:
                        n_referencias += 1
            
            referencias_versao.append(n_referencias)
        dict_ref[nome_filme] = n_referencias

    print(referencias_versao)
    print(dict_ref)

if __name__ == "__main__":
    main()

# referências inglês (MAX: Pulp Fiction)
# [4, 0, 3, 13, 1, 8, 4, 6, 9, 1, 7, 13, 2, 4, 1, 6, 2, 5, 13, 2, 1, 3, 1, 4, 2, 1, 5, 2, 2, 1, 3, 1, 1, 5, 4, 3, 3, 0, 5, 5, 11, 2, 6, 2, 16, 3, 1, 0, 3, 1, 0, 10, 4, 4, 3, 9, 1, 7, 17, 10, 4, 2, 2, 1, 5, 1, 0, 1, 4, 2, 0, 10, 9, 11, 3, 8, 2, 0, 1, 0, 3, 2, 3, 4, 2, 0, 5, 4, 4, 8, 3, 3, 1, 1, 4, 7, 1, 5, 1, 1, 1, 1, 0, 15, 1, 16, 16, 0, 2, 1, 15, 2, 12, 1, 9, 0, 6, 2, 3, 7, 4, 1, 1, 1, 3, 0, 1, 0, 1, 7, 2, 5, 2, 3, 4, 1, 1, 6, 5, 4, 0, 1, 8, 4, 3, 1, 0, 0, 6, 6, 1, 2, 0, 2, 3, 8, 7, 1, 5, 3, 2, 9, 3, 3, 4, 0, 0, 3, 7, 1, 4, 0, 2, 7, 4, 0, 1, 2, 3, 3, 2, 3, 3, 5, 2, 4, 4, 3, 4, 6, 1, 3, 1, 1, 5, 5, 7, 1, 2, 2, 1, 8, 1, 3, 1, 0, 1, 0, 3, 0, 4, 3, 2, 3, 2, 3, 2, 3, 2, 2, 2, 1, 1, 2, 4, 13, 2, 3, 5, 0, 0, 3, 2, 3, 12, 1, 3, 6, 1, 3, 1, 5, 0, 2, 2, 1, 1, 3, 4, 2, 0, 0, 4, 0, 1, 1, 13, 2, 10, 9, 3, 2, 0, 8, 1, 1, 6, 1, 2, 1, 0, 1, 2, 1, 6, 3, 2, 6, 1, 0, 11, 0, 8, 2, 3, 1, 1, 1, 1, 8, 5, 1, 1, 3, 0, 1, 0, 0, 3, 0, 1, 0, 6, 6, 0, 3, 2, 2, 0, 0, 1, 1, 3, 9, 1, 3, 3, 1, 0, 0, 0, 0, 6, 3, 1, 0, 1, 4, 2, 5, 0, 2, 2, 6, 4, 3, 3, 4, 2, 11, 2, 1, 0, 6, 0, 2, 6, 0, 2, 0, 4, 2, 6, 4, 2, 0, 4, 2, 1, 0, 3, 3, 2, 3, 1, 1, 3, 1, 2, 0, 2, 3, 4, 0, 1, 6, 3, 0, 0, 2, 1, 1, 2, 4, 3, 6, 1, 0, 0, 0, 6, 4, 0, 2, 2, 1, 2, 3, 4, 1, 4, 0, 5, 0, 1, 1, 4, 6, 0, 2, 1, 0, 1, 1, 3, 0, 0, 2, 1, 2, 2, 4, 0, 1, 1, 0, 4, 4, 1, 0, 3, 3, 4, 0, 2, 0, 2, 0, 0, 2, 0, 1, 1, 1, 2, 0, 1, 6, 5, 1, 3, 1, 2, 2, 0, 1, 1, 0, 1, 1, 1, 9, 1, 0, 3, 2, 1, 0, 2, 0, 1, 3, 1, 0, 0, 3, 1, 0, 2, 6, 1, 1, 0, 0, 1, 1, 0, 3, 1, 1, 0, 1, 0, 2]

# referências espanhol (MAX: Citizen Kane)
# [1, 0, 1, 2, 1, 3, 1, 2, 1, 0, 2, 1, 2, 2, 1, 0, 1, 1, 1, 1, 2, 0, 0, 1, 0, 1, 1, 0, 3, 2, 1, 4, 1, 0, 1, 0, 3, 3, 4, 3, 1, 0, 4, 0, 0, 0, 4, 1, 1, 1, 0, 3, 2, 1, 1, 3, 2, 0, 1, 2, 0, 0, 1, 7, 2, 0, 1, 1, 1, 2, 0, 3, 3, 0, 6, 1, 0, 0, 5, 0, 0, 1, 0, 1, 3, 4, 0, 1, 1, 1, 0, 1, 3, 0, 2, 3, 2, 1, 1, 4, 0, 1, 5, 3, 2, 0, 0, 1, 0, 0, 1, 0, 1, 2, 2, 1, 2, 0, 1, 3, 1, 1, 2, 0, 1, 1, 4, 1, 0, 0, 1, 2, 2, 1, 2, 1, 1, 3, 1, 1, 3, 0, 1, 1, 2, 1, 2, 1, 3, 4, 1, 4, 1, 0, 2, 1, 3, 2, 0, 2, 0, 0, 1, 0, 1, 2, 0, 1, 1, 1, 2, 0, 1, 1, 2, 4, 0, 0, 2, 1, 1, 1, 0, 1, 2, 0, 1, 1, 1, 2, 0, 3, 1, 2, 2, 4, 6, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 5, 0, 3, 1, 2, 1, 3, 0, 1, 1, 2, 0, 2, 0, 1, 1, 1, 0, 1, 1, 1, 3, 0, 1, 0, 1, 0, 2, 1, 0, 3, 1, 0, 2, 0, 1, 0, 1, 1, 1, 1, 1, 3, 0, 2, 2, 2, 1, 2, 0, 1, 1, 1]
    
# referências português (MAX: CISNE NEGRO)
# [2, 0, 1, 5, 6, 1, 0, 0, 1, 1, 0, 1, 0, 5, 1, 1, 2, 0, 0, 1, 0, 2, 0, 7, 1, 0, 1, 0, 2, 5, 13, 0, 7, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 1, 1, 1, 1, 2, 12, 1, 3, 1, 0, 1, 0, 1, 0, 2, 0, 0, 2, 0, 2, 0, 0, 6, 3, 6, 1, 0, 1, 0, 2, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 2, 0, 1, 0, 1, 0, 1, 0, 2, 0, 3, 1, 0, 1, 1, 1, 1, 0, 1, 5, 1, 0, 0, 9, 1, 0, 1, 0, 3, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 3, 0, 0, 4, 0, 0, 0, 1, 2, 0, 1, 2, 1, 2, 0, 7, 0, 0, 0, 0, 0, 3, 1, 0, 1, 5, 0, 0, 4, 0, 0, 1, 0, 1, 1, 0, 2, 0, 1, 1, 0, 1, 1, 1, 0, 8, 2, 4, 0, 1, 1, 0, 3, 1, 3, 4, 1, 0, 0, 0, 0, 1, 1, 0, 3, 1, 0, 2, 0, 1, 1, 1, 1, 4, 0, 1, 0, 0, 2, 6, 0, 0, 0, 0, 2, 1, 1, 0, 7, 0, 0, 0, 2, 0, 2, 4, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 5, 0, 4, 0, 0, 1, 0, 0, 1, 0, 1, 1, 3, 0, 1, 0, 1, 0, 1, 0, 0, 4, 0, 1, 0, 1, 0, 2, 7, 0]