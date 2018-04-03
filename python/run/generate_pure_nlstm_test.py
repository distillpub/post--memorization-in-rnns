
import numpy as np
import tensorflow as tf

from python.dataset import Generate
from python.model import PureNLSTM

tf.logging.set_verbosity(tf.logging.INFO)

train_dataset = Generate(repeat=True, batch_size=64)
test_dataset = Generate(dataset='test', repeat=False, batch_size=1)
model = PureNLSTM(train_dataset, name='generate_nlstm_1200',
                  embedding_size=1200,
                  verbose=True)
print(model.evaluate(dataset=test_dataset))
