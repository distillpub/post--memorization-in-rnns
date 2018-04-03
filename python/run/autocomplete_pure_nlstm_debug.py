
import json
import numpy as np
import os.path as path
import tensorflow as tf

from python.dataset import AutoComplete, AutoCompleteFixed
from python.model import PureNLSTM

dirname = path.dirname(path.realpath(__file__))
article_dir = path.join(dirname, '..', '..', 'article')

tf.logging.set_verbosity(tf.logging.INFO)

train_dataset = AutoComplete(repeat=True)
test_dataset = AutoCompleteFixed(
    "parts of north africa",
    batch_size=1,
)
model = PureNLSTM(train_dataset, name='autocomplete_nlstm_600',
                  embedding_size=600,
                  verbose=True)

for output_i, output in enumerate(
    model.predict(dataset=test_dataset)
):
    probabilities = output['probabilities']
    predict_sorted = np.argsort(probabilities, axis=1)[:, ::-1]

    source = test_dataset.decode_source(output['source'])
    target = test_dataset.decode_target(output['target'])
    predict = test_dataset.decode_target(predict_sorted)

    print(f'sequence {output_i}')
    for char, words_sorted, target_word, p in zip(source, predict, target, probabilities):
        print(f' {char} -> {words_sorted[0]}, {words_sorted[1]},'
              f' {words_sorted[2]}'
              f' -- {target_word}')
        print(p)

    break
