import torch
import torch.optim as optim
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.training.util import evaluate

from ssj500kDS import SSJ500KReader
from models import get_model, save_model_and_vocab

torch.manual_seed(1)


def train_model(parameters, name):
    token_indexer = {"tokens": ELMoTokenCharactersIndexer()} if parameters['use_elmo'] else None
    reader = SSJ500KReader(token_indexer)
    train_dataset = reader.read("train")
    validation_dataset = reader.read("test")
    vocab = Vocabulary.from_instances(train_dataset + validation_dataset)
    # vocab = Vocabulary() if parameters['use_elmo'] else Vocabulary.from_instances(train_dataset + validation_dataset)
    model = get_model(vocab, parameters)
    if torch.cuda.is_available():
        cuda_device = 0
        model = model.cuda(cuda_device)
    else:
        cuda_device = -1
    optimizer = optim.Adam(model.parameters(), lr=parameters['lr'], weight_decay=parameters['weight_decay'])
    iterator = BucketIterator(batch_size=parameters['batch_size'], sorting_keys=[("sentence", "num_tokens")])
    iterator.index_with(vocab)
    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=validation_dataset,
                      patience=parameters['patience'],
                      num_epochs=parameters['num_epochs'],
                      cuda_device=cuda_device)
    trainer.train()
    metrics = evaluate(model, validation_dataset, iterator, cuda_device, None)
    save_model_and_vocab(model, vocab, metrics, parameters, fname=name)


if __name__ == '__main__':
    parameters = {"num_epochs": 50,
                  "patience": 15,
                  "lr": 0.001,
                  "weight_decay": 1e-4,
                  "batch_size": 4,
                  "embedding_dim": 64,
                  "hidden_dim": 64,
                  "num_layers": 2,
                  "bidirectional": True,
                  "use_elmo": False,
                  "use_lstm": False}
    for bi in [True, False]:
        for lstm in [True, False]:
            for elmo in [True, False]:
                parameters["bidirectional"] = bi
                parameters["use_lstm"] = lstm
                parameters["use_elmo"] = elmo
                name = f"{'elmo' if elmo else 'embd'}_{'bi' if bi else 'uni'}_{'lstm' if lstm else 'gru'}"
                train_model(parameters, name)
