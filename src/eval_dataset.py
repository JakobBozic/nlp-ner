import torch
import numpy as np
from allennlp.predictors import SentenceTaggerPredictor
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.data.iterators import BucketIterator

from ssj500kDS import SSJ500KReader
from models import load_model_and_vocab
from sentiCorefDs import SentiCorefReader
from allennlp.training.util import evaluate

torch.manual_seed(1)


if __name__ == '__main__':
    model_dir = "elmo_bi_gru"
    # model_dir = "senti_elmo_bi_gru"
    model, params, vocab = load_model_and_vocab(model_dir)
    token_indexer = {"tokens": ELMoTokenCharactersIndexer()} if params['use_elmo'] else None
    if torch.cuda.is_available():
        cuda_device = 0
        model = model.cuda(cuda_device)
    else:
        cuda_device = -1
    iterator = BucketIterator(batch_size=params['batch_size'], sorting_keys=[("sentence", "num_tokens")])
    iterator.index_with(vocab)

    reader = SentiCorefReader(token_indexer)
    # reader = SSJ500KReader(token_indexer)
    dataset = reader.read("full")

    metrics = evaluate(model, dataset, iterator, cuda_device, None)
