
import tensorflow as tf


def embedding_matrix(vocab_size: int, dim: int,
                     name: str=None):
    with tf.name_scope(None, 'embedding-matrix'):
        # compute initialization paramters
        shape = (vocab_size - 1, dim)
        scale = tf.sqrt(1 / shape[0])

        # get or initialize embedding matrix
        w = tf.get_variable(
            name, shape,
            dtype=tf.float32,
            initializer=tf.random_uniform_initializer(
                minval=-scale, maxval=scale
            ),
            trainable=True
        )

        # 1st row should be zero and not be updated by backprop because of
        # zero padding.
        emb = tf.concat([
            tf.zeros((1, dim), dtype=tf.float32),
            w
        ], 0)

        return emb
