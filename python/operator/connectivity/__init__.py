
import tensorflow as tf

from python.operator.select import select_dim_value


def connectivity(logits, target, embedding, embedding_matrix, offset):
    logits_correct = select_dim_value(logits, target)
    # Compute partial gradient with respect to the embedding
    partial_gradient = tf.gradients(
        logits_correct[0, offset[0]],
        embedding
    )[0][0, ...]
    # Finailize the chain rule and compute the gradient with respect
    # to the one-hot-encoding of the source. Note that the
    # one-hot-encoding is not part of the graph, which is why the
    # gradient can't be computed directly this way.
    full_gradient = tf.matmul(partial_gradient,
                              tf.transpose(embedding_matrix))

    connectivity = tf.reduce_sum(full_gradient ** 2, axis=1)
    return tf.reshape(connectivity, [1, -1])
