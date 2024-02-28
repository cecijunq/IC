import wikipedia as wp

wp.set_lang("es")

def get_html():
    # esse arquivo contém os nomes dos filmes e os números de links, references e o tamanho do content de cada filme
    with open("./CSV METADADOS/aux.csv", "r") as file:
        next(file) # pula a linha que contém o cabeçalho do arquivo .csv
        for line in file: # percorre cada linha do arquivo
            vector_data_csv = line.split(';')
            nome_filme = vector_data_csv[0]

            filme = wp.page(title=nome_filme, auto_suggest=False)
            if nome_filme == "Frost/Nixon":
                nome_filme = "Frost-Nixon"
            with open('ARTIGOS HTML/espanhol/'+nome_filme+'.html','w') as f_out:
                f_out.write(filme.html())

def main():
    get_html()

if __name__ == "__main__":
    main()
