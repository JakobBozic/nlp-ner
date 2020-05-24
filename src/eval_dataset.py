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

EVAL_SSJ500K_MODEL_ON_SENTICOREF = False

# %%
if __name__ == '__main__':

    model_dir = "elmo_bi_gru" if EVAL_SSJ500K_MODEL_ON_SENTICOREF else "senti_elmo_uni_gru"

    model, params, vocab = load_model_and_vocab(model_dir)
    token_indexer = {"tokens": ELMoTokenCharactersIndexer()} if params['use_elmo'] else None
    if torch.cuda.is_available():
        cuda_device = 0
        model = model.cuda(cuda_device)
    else:
        cuda_device = -1
    iterator = BucketIterator(batch_size=params['batch_size'], sorting_keys=[("sentence", "num_tokens")])
    iterator.index_with(vocab)

    reader = SentiCorefReader(token_indexer) if EVAL_SSJ500K_MODEL_ON_SENTICOREF else SSJ500KReader(token_indexer)
    dataset = reader.read("full") if EVAL_SSJ500K_MODEL_ON_SENTICOREF else reader.read("no_misc")

    metrics = evaluate(model, dataset, iterator, cuda_device, None)
    for k, v in metrics.items():
        print(f"{k}:{v}")