import numpy as np
import matplotlib.pyplot as plt

def valuelabel(decada,links):
    for i in range(len(decada)):
        plt.text(i,links[i],links[i], ha = 'center')

n_links = []
n_references = []
n_content = []
total = 0

with open("./CSV POR DÉCADA/por década pt/30s.csv", "r") as file30s:
    soma_links = 0
    soma_references = 0
    soma_content = 0
    for line in file30s:
        vector_data_csv = line.split(';')
        nome_filme = vector_data_csv[0]
        num_links = int(vector_data_csv[1])
        num_references = int(vector_data_csv[2])
        #print(vector_data_csv[3])
        num_content = int(vector_data_csv[3])

        soma_links += num_links
        soma_references += num_references
        soma_content += num_content
        total += num_content

    n_links.append(soma_links)
    n_references.append(soma_references)
    n_content.append(soma_content)


with open("./CSV POR DÉCADA/por década pt/40s.csv", "r") as file40s:
    soma_links = 0
    soma_references = 0
    soma_content = 0
    for line in file40s:
        vector_data_csv = line.split(';')
        nome_filme = vector_data_csv[0]
        num_links = int(vector_data_csv[1])
        num_references = int(vector_data_csv[2])
        num_content = int(vector_data_csv[3])

        soma_links += num_links
        soma_references += num_references
        soma_content += num_content
        total += num_content

    n_links.append(soma_links)
    n_references.append(soma_references)
    n_content.append(soma_content)


with open("./CSV POR DÉCADA/por década pt/50s.csv", "r") as file50s:
    soma_links = 0
    soma_references = 0
    soma_content = 0
    for line in file50s:
        vector_data_csv = line.split(';')
        nome_filme = vector_data_csv[0]
        num_links = int(vector_data_csv[1])
        num_references = int(vector_data_csv[2])
        num_content = int(vector_data_csv[3])

        soma_links += num_links
        soma_references += num_references
        soma_content += num_content
        total += num_content

    n_links.append(soma_links)
    n_references.append(soma_references)
    n_content.append(soma_content)


with open("./CSV POR DÉCADA/por década pt/60s.csv", "r") as file60s:
    soma_links = 0
    soma_references = 0
    soma_content = 0
    for line in file60s:
        vector_data_csv = line.split(';')
        nome_filme = vector_data_csv[0]
        num_links = int(vector_data_csv[1])
        num_references = int(vector_data_csv[2])
        num_content = int(vector_data_csv[3])

        soma_links += num_links
        soma_references += num_references
        soma_content += num_content
        total += num_content

    n_links.append(soma_links)
    n_references.append(soma_references)
    n_content.append(soma_content)


with open("./CSV POR DÉCADA/por década pt/70s.csv", "r") as file70s:
    soma_links = 0
    soma_references = 0
    soma_content = 0
    for line in file70s:
        vector_data_csv = line.split(';')
        nome_filme = vector_data_csv[0]
        num_links = int(vector_data_csv[1])
        num_references = int(vector_data_csv[2])
        num_content = int(vector_data_csv[3])

        soma_links += num_links
        soma_references += num_references
        soma_content += num_content
        total += num_content

    n_links.append(soma_links)
    n_references.append(soma_references)
    n_content.append(soma_content)


with open("./CSV POR DÉCADA/por década pt/80s.csv", "r") as file80s:
    soma_links = 0
    soma_references = 0
    soma_content = 0
    for line in file80s:
        vector_data_csv = line.split(';')
        nome_filme = vector_data_csv[0]
        num_links = int(vector_data_csv[1])
        num_references = int(vector_data_csv[2])
        num_content = int(vector_data_csv[3])

        soma_links += num_links
        soma_references += num_references
        soma_content += num_content
        total += num_content

    n_links.append(soma_links)
    n_references.append(soma_references)
    n_content.append(soma_content)


with open("./CSV POR DÉCADA/por década pt/90s.csv", "r") as file90s:
    soma_links = 0
    soma_references = 0
    soma_content = 0
    for line in file90s:
        vector_data_csv = line.split(';')
        nome_filme = vector_data_csv[0]
        num_links = int(vector_data_csv[1])
        num_references = int(vector_data_csv[2])
        num_content = int(vector_data_csv[3])

        soma_links += num_links
        soma_references += num_references
        soma_content += num_content
        total += num_content

    n_links.append(soma_links)
    n_references.append(soma_references)
    n_content.append(soma_content)


with open("./CSV POR DÉCADA/por década pt/2000s.csv", "r") as file2000s:
    soma_links = 0
    soma_references = 0
    soma_content = 0
    for line in file2000s:
        vector_data_csv = line.split(';')
        nome_filme = vector_data_csv[0]
        num_links = int(vector_data_csv[1])
        num_references = int(vector_data_csv[2])
        num_content = int(vector_data_csv[3])

        soma_links += num_links
        soma_references += num_references
        soma_content += num_content
        total += num_content

    n_links.append(soma_links)
    n_references.append(soma_references)
    n_content.append(soma_content)


with open("./CSV POR DÉCADA/por década pt/2010s.csv", "r") as file2010s:
    soma_links = 0
    soma_references = 0
    soma_content = 0
    for line in file2010s:
        vector_data_csv = line.split(';')
        nome_filme = vector_data_csv[0]
        num_links = int(vector_data_csv[1])
        num_references = int(vector_data_csv[2])
        num_content = int(vector_data_csv[3])

        soma_links += num_links
        soma_references += num_references
        soma_content += num_content
        total += num_content

    n_links.append(soma_links)
    n_references.append(soma_references)
    n_content.append(soma_content)


with open("./CSV POR DÉCADA/por década pt/2020s.csv", "r") as file2020s:
    soma_links = 0
    soma_references = 0
    soma_content = 0
    for line in file2020s:
        vector_data_csv = line.split(';')
        nome_filme = vector_data_csv[0]
        num_links = int(vector_data_csv[1])
        num_references = int(vector_data_csv[2])
        num_content = int(vector_data_csv[3])

        soma_links += num_links
        soma_references += num_references
        soma_content += num_content
        total += num_content

    n_links.append(soma_links)
    n_references.append(soma_references)
    n_content.append(soma_content)

decadas = ['30s', '40s', '50s', '60s', '70s', '80s', '90s', '2000s', '2010s', '2020s']
contents = []

for content in n_content:
    #content = content/total
    contents.append(content)
print(total)
plt.bar(decadas, contents)
valuelabel(decadas, contents)
        
plt.xticks(va='bottom', fontsize="small")
#plt.yticks(np.arange(0, 0.5, 0.02))
plt.xlabel("Décadas")
plt.ylabel("Número de contents")
plt.title("Soma da quantidade de contents presentes por década")
plt.show()