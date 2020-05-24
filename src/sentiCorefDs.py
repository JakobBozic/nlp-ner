from bs4 import BeautifulSoup
from allennlp.data.dataset_readers import DatasetReader
from typing import Iterator, List, Dict
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data import Instance
import pickle
import random
import os

DATA_DIR = "/home/jakob/PycharmProjects/nlp-ner2/data/senticoref"

PICKLE_DUMP_FULL = "/home/jakob/PycharmProjects/nlp-ner2/data/senti_full_ds.pkl"

PICKLE_DUMP_TRAIN = "/home/jakob/PycharmProjects/nlp-ner2/data/senti_train_ds.pkl"
PICKLE_DUMP_TEST = "/home/jakob/PycharmProjects/nlp-ner2/data/senti_test_ds.pkl"

TRAIN_RATIO = 0.8


# categories: other, loc, per (merged with deriv-per), org, misc
# %%

class SentiCorefReader(DatasetReader):

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
        elif kind == "full":
            fn = PICKLE_DUMP_FULL
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


def load_convert_save_oversample(remove_in_sentence_delimeters=True):
    all_sentences = []

    for tsv in os.listdir(DATA_DIR):
        tsv_path = os.path.join(DATA_DIR, tsv)
        with open(tsv_path, "r") as tsv_file:
            lines = tsv_file.readlines()
        lines = lines[7:]
        lines_split = [line.strip("\t\n").split("\t") for line in lines]
        word_tuples = list(map(lambda x: (x[2], x[3], tsv), lines_split))
        if remove_in_sentence_delimeters:
            word_tuples = list(filter(lambda x: x[0] not in [",", ":", ";", "«", "»", "\"", "(", ")"], word_tuples))

        sentence = []
        # prev_word = word_tuples[0][0]

        for w, e, t in word_tuples:
            if w in [".", "!", "?"]:
                all_sentences.append(sentence)
                sentence = []
            else:
                sentence.append((w, e, t))

    all_tags = set()
    for sentence in all_sentences:
        for _, e, _ in sentence:
            all_tags.add(e[:3])

    data_none = []
    data_loc = []
    data_per = []
    data_org = []

    for s in all_sentences:
        if len(s) < 2:
            continue
        sentence = []
        labels = []
        lemmas = []
        all_ne_types = set()
        for w, ned, _ in s:
            word = w
            lab = "other" if ned == "-" else ned[:3].lower()
            all_ne_types.add(lab)
            sentence.append(word)
            labels.append(lab)
            lemmas.append("?")

        possible_lists = []
        if "loc" in all_ne_types:
            possible_lists.append(data_loc)
        if "org" in all_ne_types:
            possible_lists.append(data_org)
        if "per" in all_ne_types:
            possible_lists.append(data_per)
        if len(all_ne_types) == 1:
            possible_lists.append(data_none)

        list_to_add = (sorted([(len(l), l) for l in possible_lists], key=lambda x: x[0]))[0][1]

        list_to_add.append((sentence, lemmas, labels))

    max_num = max(len(data_loc), len(data_per), len(data_org), len(data_none))
    max_num_train = int(max_num * TRAIN_RATIO)

    data_train = []
    data_test = []
    for l in (data_loc, data_org, data_per, data_none):
        length = len(l)
        train_num = int(length * TRAIN_RATIO)
        for i in range(int(max_num_train / train_num)):
            data_train.extend(l[:train_num])
        data_test.extend(l[train_num:])

    with open(PICKLE_DUMP_TRAIN, "wb+") as f:
        pickle.dump(data_train, f)
    with open(PICKLE_DUMP_TEST, "wb+") as f:
        pickle.dump(data_test, f)


def load_convert_save_full():
    all_sentences = []

    for tsv in os.listdir(DATA_DIR):
        tsv_path = os.path.join(DATA_DIR, tsv)
        with open(tsv_path, "r") as tsv_file:
            lines = tsv_file.readlines()
        lines = lines[7:]
        lines_split = [line.strip("\t\n").split("\t") for line in lines]
        word_tuples = list(map(lambda x: (x[2], x[3], tsv), lines_split))
        word_tuples = list(filter(lambda x: x[0] not in [",", ":", ";", "«", "»", "\"", "(", ")"], word_tuples))

        sentence = []
        # prev_word = word_tuples[0][0]

        for w, e, t in word_tuples:
            if w in [".", "!", "?"]:
                all_sentences.append(sentence)
                sentence = []
            else:
                sentence.append((w, e, t))

    # all_tags = set()
    # for sentence in all_sentences:
    #     for _, e, _ in sentence:
    #         all_tags.add(e[:3])

    data = []

    for s in all_sentences:
        if len(s) < 2:
            continue
        sentence = []
        labels = []
        lemmas = []
        all_ne_types = set()
        for w, ned, _ in s:
            word = w
            lab = "other" if ned == "-" else ned[:3].lower()
            all_ne_types.add(lab)
            sentence.append(word)
            labels.append(lab)
            lemmas.append("?")

        data.append((sentence, lemmas, labels))

    with open(PICKLE_DUMP_FULL, "wb+") as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    random.seed(1337)
    load_convert_save_full()
