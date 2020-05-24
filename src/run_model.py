import torch
import numpy as np
from allennlp.predictors import SentenceTaggerPredictor
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer

from ssj500kDS import SSJ500KReader
from models import load_model_and_vocab

torch.manual_seed(1)


# fname = "04_30_00_06"


def run_text_input(model_dir, sentence):
    model, params, _ = load_model_and_vocab(model_dir)
    token_indexer = {"tokens": ELMoTokenCharactersIndexer()} if params['use_elmo'] else None
    reader = SSJ500KReader(token_indexer)
    predictor = SentenceTaggerPredictor(model, dataset_reader=reader)
    tag_logits = predictor.predict(sentence)['tag_logits']
    tag_ids = np.argmax(tag_logits, axis=-1)
    print([(w, model.vocab.get_token_from_index(i, 'labels')) for w, i in zip(sentence.split(" "), tag_ids)])


if __name__ == '__main__':
    sentence = "Po različnih delih Slovenije zaznavamo od 10 do 30 odstotkov manj rakavih diagnoz Kako bo to vplivalo na dolgi rok ta trenutek še ne vemo je za Televizijo Slovenija povedala Vesna Zadnik vodja epidemiologije in registra raka na Onkološkem inštitutu"
    model_dir = "elmo_bi_gru"

    run_text_input(model_dir, sentence)
