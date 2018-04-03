
import os.path as path

import tensorflow as tf

from python.operator import embedding_matrix
from python.model.abstact import AbstactModel

dirname = path.dirname(path.realpath(__file__))


class PassThrough(AbstactModel):
    def __init__(self, dataset, name='pass_through', **kwargs):
        super().__init__(
            dataset,
            params={
                'source_classes': dataset.source_classes,
                'target_classes': dataset.target_classes
            },
            name=name,
            **kwargs
        )

    def _model_fn(self, features, labels, mode, params):
        source = features['source']
        length = features['length']
        # If `mode == PREDICT` labels will be None, to make the graph
        # consistent for all models, just default it to an zero tensor.
        target = tf.zeros_like(source) if labels is None else labels

        # Create logistic regression, because apparently a valid train
        # model must exists.
        source_embedding = embedding_matrix(
            vocab_size=params['source_classes'],
            dim=params['target_classes'],
            name='source-embedding'
        )

        # lookup embedding
        logits = tf.nn.embedding_lookup(source_embedding, source)

        # compute prediction
        predict = tf.argmax(logits, 2)

        # compute losses
        loss = tf.losses.sparse_softmax_cross_entropy(
            logits=logits,
            labels=target,
            # mask <eos> out of losses
            weights=tf.not_equal(target, tf.zeros_like(target)),
            reduction=tf.losses.Reduction.MEAN
        )

        # Create optimizer
        optimizer = tf.train.AdagradOptimizer(learning_rate=1)
        train_op = optimizer.minimize(
            loss,
            global_step=tf.train.get_global_step()
        )

        # Return EstimatorSpec, depending on the mode tensorflow will
        # automatically select the parameters that are required.
        return tf.estimator.EstimatorSpec(
            mode,
            loss=loss,
            train_op=train_op,
            predictions={
                'length': length,
                'source': source,
                'target': features['target'],

                'logits': logits,
                'probabilities': tf.nn.softmax(logits),
                'predict': predict
            },
            eval_metric_ops={
                'accuracy': tf.metrics.accuracy(labels=target,
                                                predictions=predict),
                'cross_entropy': tf.metrics.mean(loss)
            }
        )
