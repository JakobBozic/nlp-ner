from bs4 import BeautifulSoup
from allennlp.data.dataset_readers import DatasetReader
from typing import Iterator, List, Dict
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data import Instance
import pickle
import random

PICKLE_DUMP_FULL = "/home/jakob/PycharmProjects/nlp-ner/data/full_ds.pkl"

PICKLE_DUMP_TRAIN = "/home/jakob/PycharmProjects/nlp-ner/data/train_ds.pkl"
PICKLE_DUMP_TEST = "/home/jakob/PycharmProjects/nlp-ner/data/test_ds.pkl"
PICKLE_DUMP_NO_MISC = "/home/jakob/PycharmProjects/nlp-ner/data/ssj500k_no_misc_full.pkl"

TRAIN_RATIO = 0.8


# categories: other, loc, per (merged with deriv-per), org, misc


class SSJ500KReader(DatasetReader):

    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def text_to_instance(self, tokens: List[Token], tags: List[str] = None) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"sentence": sentence_field}

        if tags:
            label_field = SequenceLabelField(labels=tags, sequence_field=sentence_field)
            fields["labels"] = label_field

        return Instance(fields)

    def _read(self, kind) -> Iterator[Instance]:
        if kind == "train":
            fn = PICKLE_DUMP_TRAIN
        elif kind == "test":
            fn = PICKLE_DUMP_TEST
        elif kind == "no_misc":
            fn = PICKLE_DUMP_NO_MISC
        else:
            raise Exception(f"Uknown kind {kind}")
        with open(fn, "rb") as f:
            data = pickle.load(f)
            for words, lemmas, tags in data:
                if len(words) > 0:
                    sentence_field = TextField([Token(word) for word in words], self.token_indexers)
                    fields = {"sentence": sentence_field}
                    label_field = SequenceLabelField(labels=tags, sequence_field=sentence_field)
                    fields["labels"] = label_field
                    yield Instance(fields)


def load_convert_save():
    tei_doc = "/home/jakob/PycharmProjects/nlp-ner/data/ssj500k-sl.TEI/ssj500k-sl.body.xml"
    with open(tei_doc, 'r') as tei:
        soup = BeautifulSoup(tei, 'lxml')

    all_sentences = soup.findAll("s")
    del soup

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
    with open(PICKLE_DUMP_FULL, "wb+") as f:
        pickle.dump(data, f)


def load_convert_save_oversample():
    tei_doc = "/home/jakob/PycharmProjects/nlp-ner/data/ssj500k-sl.TEI/ssj500k-sl.body.xml"
    with open(tei_doc, 'r') as tei:
        soup = BeautifulSoup(tei, 'lxml')

    all_sentences = soup.findAll("s")
    random.shuffle(all_sentences)

    num_all = len(all_sentences)

    del soup

    data_none = []
    data_loc = []
    data_per = []
    data_org = []
    data_misc = []

    for s in all_sentences:
        sentence = []
        labels = []
        lemmas = []
        words = s.findAll("w")
        ned = {}
        nel = s.findAll("seg")
        for ne in nel:
            for w in ne.findAll("w"):
                subtype = ne.get("subtype")
                if subtype == "deriv-per":
                    subtype = "per"
                ned[w.text] = subtype
        for w in words:
            word = w.text
            lab = ned[word] if word in ned else "other"
            sentence.append(word)
            labels.append(lab)
            lemmas.append(w.get("lemma"))
        all_ne_types = set(ned.values())
        if "loc" in all_ne_types:
            data_loc.append((sentence, lemmas, labels))
        if "per" in all_ne_types or "deriv-per" in all_ne_types:
            data_per.append((sentence, lemmas, labels))
        if "org" in all_ne_types:
            data_org.append((sentence, lemmas, labels))
        if "misc" in all_ne_types:
            data_misc.append((sentence, lemmas, labels))
        if len(all_ne_types) == 0:
            data_none.append((sentence, lemmas, labels))

    max_num = max(len(data_loc), len(data_per), len(data_org), len(data_misc))
    max_num_train = int(max_num * TRAIN_RATIO)

    data_train = []
    data_test = []
    for l in (data_loc, data_org, data_per, data_misc):
        length = len(l)
        train_num = int(length * TRAIN_RATIO)
        for i in range(int(max_num_train / train_num)):
            data_train.extend(l[:train_num])
        data_test.extend(l[train_num:])

    with open(PICKLE_DUMP_TRAIN, "wb+") as f:
        pickle.dump(data_train, f)
    with open(PICKLE_DUMP_TEST, "wb+") as f:
        pickle.dump(data_test, f)


def load_convert_save_no_misc():
    tei_doc = "/home/jakob/PycharmProjects/nlp-ner/data/ssj500k-sl.TEI/ssj500k-sl.body.xml"
    with open(tei_doc, 'r') as tei:
        soup = BeautifulSoup(tei, 'lxml')

    all_sentences = soup.findAll("s")
    random.shuffle(all_sentences)

    num_all = len(all_sentences)

    del soup

    data_train = []

    for s in all_sentences:
        sentence = []
        labels = []
        lemmas = []
        words = s.findAll("w")
        ned = {}
        nel = s.findAll("seg")
        for ne in nel:
            for w in ne.findAll("w"):
                subtype = ne.get("subtype")
                if subtype == "deriv-per":
                    subtype = "per"
                ned[w.text] = subtype
        for w in words:
            word = w.text
            lab = ned[word] if (word in ned and ned[word] != "misc") else "other"
            sentence.append(word)
            labels.append(lab)
            lemmas.append(w.get("lemma"))
        all_ne_types = set(ned.values())
        if len(all_ne_types) > 1:
            data_train.append((sentence, lemmas, labels))

    with open(PICKLE_DUMP_NO_MISC, "wb+") as f:
        pickle.dump(data_train, f)


if __name__ == '__main__':
    random.seed(1337)
    load_convert_save_no_misc()
