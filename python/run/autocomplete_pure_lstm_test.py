
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
print(model.evaluate(dataset=test_dataset))
