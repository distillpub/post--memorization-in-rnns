
import tensorflow as tf


def select_dim_value(x, indices, name=None):
    with tf.name_scope(name, "select-dim-value", values=[x, indices]):
        # x.shape = (rest..., dims)
        rest = tf.shape(x)[:-1]
        dims = tf.shape(x)[-1]
        size = tf.size(indices, out_type=indices.dtype)

        # reshape to (size, dims)
        t = tf.reshape(x, shape=[-1, dims])
        # then index as ([1,2,3,...,size], indices.ravel())
        nd_indices = tf.stack([
            tf.range(0, size, dtype=indices.dtype),
            tf.reshape(indices, shape=[-1])
        ], axis=1)
        t = tf.gather_nd(t, indices=nd_indices)

        # reshape back to (rest...)
        t = tf.reshape(t, rest)
        t.set_shape(x.get_shape()[:-1])
        return t
