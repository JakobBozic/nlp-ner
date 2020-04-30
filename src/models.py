from typing import Iterator, List, Dict
import torch
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.predictors import SentenceTaggerPredictor
from allennlp.modules.token_embedders import ElmoTokenEmbedder
from datetime import datetime
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
import os


class NerModel(Model):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.hidden2tag = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                          out_features=vocab.get_vocab_size('labels'))
        self.accuracy = CategoricalAccuracy()
        self.fms = [F1Measure(i) for i in range(1, 5)]

    def forward(self,
                sentence: Dict[str, torch.Tensor],
                labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(sentence)
        embeddings = self.word_embeddings(sentence)
        encoder_out = self.encoder(embeddings, mask)
        tag_logits = self.hidden2tag(encoder_out)
        output = {"tag_logits": tag_logits}
        if labels is not None:
            self.accuracy(tag_logits, labels, mask)
            for fm in self.fms:
                fm(tag_logits, labels, mask)
            output["loss"] = sequence_cross_entropy_with_logits(tag_logits, labels, mask)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        dct = {"accuracy": self.accuracy.get_metric(reset),
               "avg_f1": 0}
        for i, fm in enumerate(self.fms):
            p, r, f = fm.get_metric(reset)
            dct[f"p{i}"] = p
            dct[f"r{i}"] = r
            dct[f"f{i}"] = f
            dct["avg_f1"] = dct['avg_f1'] + f
        dct["avg_f1"] = dct["avg_f1"] / 4
        return dct


def get_model(vocab, params):
    options_file = "/home/jakob/PycharmProjects/nlp-ner/models/slovenian-elmo/options.json"
    weights_file = "/home/jakob/PycharmProjects/nlp-ner/models/slovenian-elmo/slovenian-elmo-weights.hdf5"

    emb_d = params["embedding_dim"]
    hidden_d = params["hidden_dim"]

    use_elmo_embeddings = params['use_elmo']
    use_lstm = params['use_lstm']
    n_layers = params["num_layers"]

    bidirectional = params['bidirectional']

    if use_elmo_embeddings:
        token_embedder = ElmoTokenEmbedder(options_file, weights_file)
    else:
        token_embedder = Embedding(num_embeddings=vocab.get_vocab_size('tokens'), embedding_dim=emb_d)

    word_embedder = BasicTextFieldEmbedder({"tokens": token_embedder})
    emb_d = word_embedder.get_output_dim()

    if use_lstm:
        encoder = PytorchSeq2SeqWrapper(torch.nn.LSTM(emb_d, hidden_d, num_layers=n_layers, batch_first=True, bidirectional=bidirectional))
    else:
        encoder = PytorchSeq2SeqWrapper(torch.nn.GRU(emb_d, hidden_d, num_layers=n_layers, batch_first=True, bidirectional=bidirectional))

    model = NerModel(word_embedder, encoder, vocab)
    return model


def save_model_and_vocab(model, vocab, metrics, params=None, fname=None):
    if fname is None:
        fname = datetime.now().strftime("%m_%d_%H_%M")
    os.makedirs(os.path.join("/home/jakob/PycharmProjects/nlp-ner/models", fname), exist_ok=True)

    with open(os.path.join("/home/jakob/PycharmProjects/nlp-ner/models", fname, "model.stdct"), 'wb') as f:
        torch.save(model.state_dict(), f)
    vocab.save_to_files(os.path.join("/home/jakob/PycharmProjects/nlp-ner/models", fname, "vocab"))
    with open(os.path.join("/home/jakob/PycharmProjects/nlp-ner/models", fname, "parameters.txt"), 'w+') as f:
        lines = sorted(map(lambda x: f"{x[0]}:{x[1]}\n", params.items()))
        f.writelines(lines)
    with open(os.path.join("/home/jakob/PycharmProjects/nlp-ner/models", fname, "metrics.txt"), 'w+') as f:
        lines = list(map(lambda x: f"{x[0]}:{x[1]}\n", metrics.items()))
        f.writelines(lines)


def load_model_and_vocab(fname):
    FOLDER_PATH = os.path.join("/home/jakob/PycharmProjects/nlp-ner/models", fname)
    MODEL_PATH = os.path.join(FOLDER_PATH, "model.stdct")
    VOCAB_PATH = os.path.join(FOLDER_PATH, "vocab")
    PARAMS_PATH = os.path.join(FOLDER_PATH, "parameters.txt")

    vocab = Vocabulary.from_files(VOCAB_PATH)
    params = parse_params(PARAMS_PATH)
    model = get_model(vocab, params)

    if torch.cuda.is_available():
        cuda_device = 0
    else:
        cuda_device = -1

    with open(MODEL_PATH, 'rb') as f:
        model.load_state_dict(torch.load(f))

    if cuda_device > -1:
        model.cuda(cuda_device)
    return model, params


def parse_params(p_fname):
    with open(p_fname, "r") as f:
        lines = f.readlines()
    params = {}
    for p in lines:
        k, v = p.strip().split(':')
        try:
            v = parse_bool(v)
        except Exception:
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except Exception:
                    pass
        params[k] = v
    return params


def parse_bool(v):
    if v.lower() == 'false':
        return False
    elif v.lower() == 'true':
        return True
    else:
        raise Exception
