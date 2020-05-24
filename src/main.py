import torch
import conllu
from tei_reader import TeiReader
import vert
from bs4 import BeautifulSoup

tei_doc = "/home/jakob/PycharmProjects/nlp-ner/data/ssj500k-sl.TEI/ssj500k-sl.body.xml"
with open(tei_doc, 'r') as tei:
    soup = BeautifulSoup(tei, 'lxml')

# with open("/home/jakob/PycharmProjects/nlp-ner/data/ssj500k/sl_ssj-ud_v2.4.conllu") as f:
#     data = conllu.parse(f.read())
#
# corpora = TeiReader().read_file("/home/jakob/PycharmProjects/nlp-ner/data/ssj500k-sl.TEI/ssj500k-sl.body.xml")
# print(corpora.text)
# %%
# first_p = soup.findChild("p")
# sentences = first_p.findAll("s")
# first_s = sentences[1]
# words = list(map(lambda x:x.text, first_s.find_all('w')))



all_sentences = soup.findAll("s")
del soup

# %%
# %%
data = []
for s in all_sentences:
    sentence = []
    labels = []
    lemmas = []
    words = s.findAll("w")
    ned = {}
    nel = s.findAll("seg")
    for ne in nel:
        for w in ne.findAll("w"):
            ned[w.text] = ne.get("subtype")
    for w in words:
        word = w.text
        lab = ned[word] if word in ned else "other"
        sentence.append(word)
        labels.append(lab)
        lemmas.append(w.get("lemma"))
    data.append((sentence, lemmas, labels))

