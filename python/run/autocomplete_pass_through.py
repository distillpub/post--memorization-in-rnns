
import numpy as np
import tensorflow as tf

from python.dataset import AutoComplete
from python.model import PassThrough

tf.logging.set_verbosity(tf.logging.INFO)

train_dataset = AutoComplete(repeat=True)
valid_dataset = AutoComplete(dataset='test', repeat=False, batch_size=1)
model = PassThrough(train_dataset,
                    name='autocomplete_passthough',
                    verbose=True)
model.train(max_steps=1)

for output_i, output in enumerate(
    model.predict(dataset=valid_dataset)
):
    length = output['length']
    source = valid_dataset.decode_source(output['source'])
    target = valid_dataset.decode_target(output['target'])

    print(f'sequence {output_i} of length {length}')
    for char, word in zip(source, target):
        print(f' {char} -> {word}')

    break
