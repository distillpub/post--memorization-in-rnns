
import tensorflow as tf

from python.operator.embedding import embedding_matrix
from python.operator.nlstm import BasicNLSTMCell


def nlstm_aligned(source: tf.Tensor, length: tf.Tensor,
                  source_vocab_size=64, target_vocab_size=64,
                  latent_dim=200, depth=2, layers=1):
    # make embedding matrix
    source_embedding = embedding_matrix(
        vocab_size=source_vocab_size,
        dim=latent_dim,
        name='embedding-source'
    )

    # lookup embedding
    embedding = tf.nn.embedding_lookup(source_embedding, source)

    # https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn
    rnn_layers = [
        BasicNLSTMCell(latent_dim, depth=depth) for i in range(layers)
    ]
    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
    rnn, state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                   inputs=embedding,
                                   sequence_length=length,
                                   dtype=tf.float32)

    # final dense layer
    dense = tf.layers.dense(rnn, target_vocab_size)

    return dense, embedding, source_embedding
