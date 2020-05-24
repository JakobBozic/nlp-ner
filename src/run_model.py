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
    sentence = "Obžalujem da se je komunikacija med Naso in Roskozmosov zadnja leta izrazito poslabšala Namesto da bi razpravljali o ducatih projektov ki so nam v skupnem interesu ostajamo samo na izstrelitvah astronavtov na Mednarodno vesoljsko postajo ali pa dobavljanju ruskih motorjev RD-180/81 v ZDA Visokoleteči projekti povezani s kolonizacijo Meseca bi lahko bili močna skupna točka obeh držav v težavnih časih"
    model_dir = "senti_elmo_bi_gru"

    run_text_input(model_dir, sentence)
