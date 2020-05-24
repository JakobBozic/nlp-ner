from allennlp.modules.elmo import Elmo, batch_to_ids

options_file = "/home/jakob/PycharmProjects/nlp-ner2/models/slovenian-elmo/options.json"
weights_file = "/home/jakob/PycharmProjects/nlp-ner2/models/slovenian-elmo/slovenian-elmo-weights.hdf5"

elmo = Elmo(options_file, weights_file, 3, dropout=0)

sentences = [['Kdo', "je", "tu", "nor", 'jebemti'], ['Prasica']]
character_ids = batch_to_ids(sentences)

embeddings = elmo(character_ids)


class ElmoSlo:
    def __init__(self) -> None:
        super().__init__()
        self.elmo = Elmo(options_file, weights_file, 1, dropout=0, requires_grad=False)

    def __call__(self, *args, **kwargs):
        return self.elmo(*args, **kwargs)


from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import ElmoTokenEmbedder

options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json'
weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5'

elmo_embedder = ElmoTokenEmbedder(options_file, weight_file)
word_embeddings = BasicTextFieldEmbedder({"tokens": elmo_embedder})