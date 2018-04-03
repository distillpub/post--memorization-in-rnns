
import numpy as np
import tensorflow as tf

from python.dataset import AutoComplete
from python.model import PureLSTM

tf.logging.set_verbosity(tf.logging.INFO)

train_dataset = AutoComplete(repeat=True, batch_size=64)
test_dataset = AutoComplete(dataset='test', repeat=False, batch_size=1)
model = PureLSTM(train_dataset, name='autocomplete_lstm_600',
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
    for char, words_sorted, target_word in zip(source, predict, target):
        print(f' {char} -> {words_sorted[0]}, {words_sorted[1]},'
              f' {words_sorted[2]}'
              f' -- {target_word}')
