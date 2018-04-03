
import json
import numpy as np
import os.path as path
import tensorflow as tf

from python.dataset import AutoComplete, AutoCompleteFixed
from python.model import PureGRU

dirname = path.dirname(path.realpath(__file__))
article_dir = path.join(dirname, '..', '..', 'article')

tf.logging.set_verbosity(tf.logging.INFO)

train_dataset = AutoComplete(repeat=True)
test_dataset = AutoCompleteFixed(
    "parts of north africa",
    batch_size=1
)
model = PureGRU(train_dataset, name='autocomplete_gru_600',
                embedding_size=600,
                verbose=True)

data = []

print(f'sequence:')
for output_i, output in enumerate(
    model.predict(dataset=test_dataset)
):
    probabilities = output['probabilities']
    source = test_dataset.decode_source(output['source'])

    for index, probs in enumerate(probabilities):
        prefix = ''.join(source[0:index + 1].tolist())
        print(f'  "{prefix}"')
        data.append([prefix, probs.tolist()])

with open(path.join(article_dir, 'data/demo–precompute.json'), 'w') as file:
    json.dump(data, file)

print('done')
