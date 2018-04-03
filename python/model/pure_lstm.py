
import os.path as path

import tensorflow as tf

from python.operator import lstm_aligned, connectivity
from python.model.abstact import AbstactModel

dirname = path.dirname(path.realpath(__file__))


class PureLSTM(AbstactModel):
    def __init__(self, dataset, name='pure_lstm',
                 embedding_size=1200, layers=2, **kwargs):
        super().__init__(
            dataset,
            params={
                'embedding_size': embedding_size,
                'layers': layers,
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

        # compute logits for model
        logits, embedding, embedding_matrix = lstm_aligned(
            source, length,
            source_vocab_size=params['source_classes'],
            target_vocab_size=params['target_classes'],
            latent_dim=params['embedding_size'],
            layers=params['layers'],
            target=features['target']
        )

        # compute prediction
        predict = tf.argmax(logits, 2)

        # compute losses
        loss = tf.losses.sparse_softmax_cross_entropy(
            logits=logits,
            labels=target,
            # mask <eos> <unknown> out of losses, (t != 0 && t != 1)
            weights=tf.logical_and(
                tf.not_equal(target, tf.zeros_like(target)),
                tf.not_equal(target, tf.ones_like(target))
            ),
            reduction=tf.losses.Reduction.MEAN
        )

        # Create optimizer
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(
            loss,
            global_step=tf.train.get_global_step()
        )

        # Compute connectivity
        connectivity_sample = tf.zeros_like(source, dtype='float32')
        if 'offset' in features:
            connectivity_sample = connectivity(
                logits, features['target'],
                embedding, embedding_matrix,
                features['offset']
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
                'predict': predict,

                'connectivity': connectivity_sample
            },
            eval_metric_ops={
                'accuracy': tf.metrics.accuracy(labels=target,
                                                predictions=predict),
                'cross_entropy': tf.metrics.mean(loss)
            }
        )
