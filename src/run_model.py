import torch
import numpy as np
from allennlp.predictors import SentenceTaggerPredictor
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer

from ssj500kDS import SSJ500KReader
from models import load_model_and_vocab

torch.manual_seed(1)

fname = "04_30_00_06"

model, params = load_model_and_vocab(fname)
token_indexer = {"tokens": ELMoTokenCharactersIndexer()} if params['use_elmo'] else None
reader = SSJ500KReader(token_indexer)
predictor = SentenceTaggerPredictor(model, dataset_reader=reader)
sentence = "Janez Ivan Marija Petrol Danes je bil v Vipavi lep sončen dan Ankaran Gorica Celje Ljubljana Google IKEA Koper trgovina FAMA je bila zaprta na sprehod sem odšel z Alo"
tag_logits = predictor.predict(sentence)['tag_logits']
tag_ids = np.argmax(tag_logits, axis=-1)
print([(w, model.vocab.get_token_from_index(i, 'labels')) for w, i in zip(sentence.split(" "), tag_ids)])
