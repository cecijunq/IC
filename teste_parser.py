import urllib.request
import wikipedia as wp
from bs4 import BeautifulSoup
import requests, re

from urllib.request import urlopen

#def removeOneTag(text, tag):
    #return text[:text.find("<"+tag+">")] + text[text.find("</"+tag+">") + len(tag)+3:]

filme = wp.page(title="Titanic (1997 film)", auto_suggest=False)
fp = urllib.request.urlopen(filme.url)
html_doc = fp.read()
mystr = html_doc.decode("utf8")

soup = BeautifulSoup(mystr, 'html.parser')
#soup = BeautifulSoup(html_doc, 'html.parser')

print(type(html_doc))
#print(type(mystr))

referenciados = []

for tag in soup.select('a:not(h2#See_also ~ a)', href=True):
    url = tag.get("href")
    print(url)
    print(type(url))
    #url = a.attrs['href']
    if url is not None:
        print("entrou")
        url = url.replace("/wiki/", "https://en.wikipedia.org/wiki/")
    #print(url)
    #soup2 = BeautifulSoup(urlopen(url))
    #f = requests.get(url)
    # displaying the title
    #print("Title of the website is : ")
    #print (soup2.title.get_text())
    #print(a.text)
    #print(a.attrs)
    referenciados.append(tag.string)


"""for a in soup.find_all('a:not(h2#See_also ~ a)', href=True):
    print("Found the URL:", a['href'])
"""

"""i = 0
for character in mystr:
    if i < len(mystr) and mystr[i:i+7] == 'id="See_also"':
        mystr[i:] = ""
    i+=1

print(mystr)"""

"""referencias_corpo = soup.select_one('div#mw-content-text')
referencias_final = soup.select_one('div#catlinks')
print(type(referencias_final))

referenciados = []
links_corpo = referencias_corpo.find_all("a")
for x in links_corpo:
    #page = x.get('href')
    referenciados.append(x.string)
    #print(x.string)

links_final = referencias_final.find_all("a")
for x in links_final:
    if x in links_corpo:
        #page = x.get('href')
        referenciados.append(x.string)
    #print(x.string)
"""

referenciados = list(dict.fromkeys(referenciados))
#print(referenciados)
fp.close()

#with open("yourhtmlfile.html", "w") as file:
    #file.write(mystr)
