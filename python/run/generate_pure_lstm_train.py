
import tensorflow as tf

from python.dataset import Generate
from python.model import PureLSTM

tf.random.set_random_seed(450849059)  # From random.org
tf.logging.set_verbosity(tf.logging.INFO)

train_dataset = Generate(repeat=True, batch_size=64)
valid_dataset = Generate(repeat=True, dataset='valid', batch_size=64)

model = PureLSTM(train_dataset, name='generate_lstm_1200',
                 embedding_size=1200,
                 verbose=True)
model.train(max_steps=train_dataset.batches, valid_dataset=valid_dataset)
