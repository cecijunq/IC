import pandas as pd
import matplotlib.pyplot as plt

# dict_freq = {}
# df = pd.read_csv("films_imdb.csv", sep=";")
# #print(df.to_string())

# for index, row in df.iterrows():
#     print(row['genres'], type(row['genres']))
#     if row['genres'] not in dict_freq.keys():
#         dict_freq[row['genres']] = 1
#     else:
#         dict_freq[row['genres']] += 1
# total = sum(dict_freq.values())

# values_dict = []
# for e in dict_freq.values():
#     values_dict.append(e/total)

# x_values = range(len(dict_freq.keys()))
# fig = plt.figure(figsize = (18, 8))
# plt.bar(x_values, values_dict, align="edge", width=0.6)
# plt.xticks(x_values, dict_freq.keys(), rotation="vertical", fontsize=7)
# plt.title("Gêneros dos filmes que concorreram à categoria Melhor Filme no Oscar (1929 - 2023) - fonte: IMDb")
# plt.show()

dict_freq = {}
df = pd.read_csv("films_imdb.csv", sep=";")

for index, row in df.iterrows():
    #print(row['genres'], type(row['genres']))
    genres = row['genres'].split(",")
    #print(genres)

    if genres[0] not in dict_freq.keys():
        dict_freq[genres[0]] = 1
    else:
        dict_freq[genres[0]] += 1
    # for genre in genres:
    #     if genre not in dict_freq.keys():
    #         dict_freq[genre] = 1
    #     else:
    #         dict_freq[genre] += 1
print(dict_freq)

total = sum(dict_freq.values())

values_dict = []
for e in dict_freq.values():
    values_dict.append(e/total)

x_values = range(len(dict_freq.keys()))
fig = plt.figure(figsize = (18, 8))
plt.bar(x_values, values_dict, align="edge", width=0.6)
plt.xticks(x_values, dict_freq.keys(), rotation="vertical", fontsize=7)
plt.title("Gêneros dos filmes que concorreram à categoria Melhor Filme no Oscar (1929 - 2023) - fonte: IMDb")
plt.show()