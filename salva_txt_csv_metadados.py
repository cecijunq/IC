import wikipedia
import difflib

import pandas as pd

#### algoritmo de similaridade 1
# Levenshtein Distance Algorithm
def levenshtein_distance(s, t):
    m, n = len(s), len(t)
    if m < n:
        s, t = t, s
        m, n = n, m
    d = [list(range(n + 1))] + [[i] + [0] * n for i in range(1, m + 1)]
    for j in range(1, n + 1):
        for i in range(1, m + 1):
            if s[i - 1] == t[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1]) + 1
    return d[m][n]
 
def compute_similarity(input_string, reference_string):
    distance = levenshtein_distance(input_string, reference_string)
    max_length = max(len(input_string), len(reference_string))
    similarity = 1 - (distance / max_length)
    return similarity


#### algoritmo de similaridade 2
def similarity_test(str1, str2):
    result =  difflib.SequenceMatcher(a=str1.lower(), b=str2.lower())
    return result.ratio()

def f(links):
    str = ''
    for link in links:
        str += link + '\n'
    
    return str

def counts_number_links(self):
    n = 0
    for link in self:
        n+=1
    return n

def main():
    wikipedia.set_lang("es")

    with open("nomes filmes pt.txt", "r") as file:
        for line in file:
            nome = line
            nome = nome.replace("\n", "")
            nome = nome.replace("_", " ")

            result = wikipedia.search(nome)

            print(result)
            if similarity_test(nome.lower(), result[1]) > 0.6:
                page_link = wikipedia.page(title=result[1], auto_suggest=False)

                with open("./ARTIGOS TXT/artigos espanhol/"+nome+".txt", 'w') as file:
                    file.write(page_link.title + '\n')
                    file.write(page_link.content)
                    file.write(f(page_link.links))
                    file.write(f(page_link.references))

                    print(counts_number_links(page_link.links))
                    print(counts_number_links(page_link.references))

                with open("./CSV METADADOS/metadados_es.csv", 'a') as file2:
                    links = counts_number_links(page_link.links)
                    references = counts_number_links(page_link.references)
                    tamanho_content = len(page_link.content)
                    file2.write("{};{};{};{}\n".format(page_link.title,links,references,tamanho_content))

def main2():
    wikipedia.set_lang("es")
    with open("nomes filmes espanhol.txt", 'r') as file:
        for line in file:
            vector_data_csv = line.split(';')
            print(vector_data_csv)

            with open("oscar.csv", "a") as csv_file:
                nome = vector_data_csv[0]
                nome = nome.replace("_", " ")
                nome = nome.replace("\n", "")
                #venceu = vector_data_csv[1]
                #ano = vector_data_csv[2]
                #ano = ano.replace("\n", "")

                result = wikipedia.search(nome)
                #print(result)

                #print(similarity_test(nome.title(), result[0]))
                #print("\n")
                if similarity_test(nome.lower(), result[0]) > 0.6:
                    page_link = wikipedia.page(title=result[0], auto_suggest=False)
                    #csv_file.write("{};{};{};{}\n".format(ano,nome,venceu,page_link.url))
                    with open("./metadados_es.csv", 'a') as file:
                        links = counts_number_links(page_link.links)
                        references = counts_number_links(page_link.references)
                        tamanho_content = len(page_link.content)
                        file.write("{};{};{};{}\n".format(page_link.title,links,references,tamanho_content))

                    # CÓDIGO QUE REGISTRA EM UM ARQUIVO TXT OS CONTEÚDO, OS LINKS E OS REFERENCES DE UMA PÁGINA DA WIKIPEDIA
                    #with open("./artigos inglês/"+nome+".txt", 'w') as file:
                        #file.write(page_link.title + '\n')
                        #file.write(page_link.content)
                        #file.write(f(page_link.links))
                        #file.write(f(page_link.references))

                        #print(counts_number_links(page_link.links))
                        #print(counts_number_links(page_link.references))
                
    
if __name__ == "__main__":
    main()