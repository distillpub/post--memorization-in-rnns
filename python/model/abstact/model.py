
import os.path as path

import tensorflow as tf

from python.operator import lstm_aligned

dirname = path.dirname(path.realpath(__file__))
save_dir = path.realpath(path.join(dirname, '..', '..', 'save'))


class InputWrapper:
    def __init__(self, train_dataset, valid_dataset):
        self._train_dataset = train_dataset
        self._valid_dataset = valid_dataset

    def __call__(self):
        features = dict()
        lables = dict()

        train_features, train_label = self._train_dataset()
        for key, value in train_features.items():
            features[f'train#{key}'] = value
        lables['train'] = train_label

        if self._valid_dataset is not None:
            valid_features, valid_label = self._valid_dataset()
            for key, value in valid_features.items():
                features[f'valid#{key}'] = value
            lables['valid'] = valid_label

        return (features, lables)


class AbstactModel:
    def __init__(self, dataset, params={}, name='unamed_model', verbose=False):
        self._verbose = verbose
        self._dataset = dataset

        # https://www.tensorflow.org/get_started/custom_estimators
        # https://www.tensorflow.org/get_started/checkpoints
        self._estimator = tf.estimator.Estimator(
            model_fn=self._root_model_fn,
            params=params,
            config=tf.estimator.RunConfig(
                # Save summary every step
                save_summary_steps=1,
                # Save checkpoints every minute.
                save_checkpoints_secs=60,
                # Retain the 5 most recent checkpoints.
                keep_checkpoint_max=5,
                model_dir=path.join(save_dir, name)
            )
        )

    def _root_model_fn(self, features, labels, mode, params):
        if mode == tf.estimator.ModeKeys.TRAIN:
            # Add training
            train_features = {
                key[6:]: value for key, value in features.items()
                if key[:6] == 'train#'
            }
            train_label = labels['train']
            with tf.variable_scope("model"):
                spec = self._model_fn(
                    train_features, train_label, mode, params
                )

            # Add validation
            if 'valid' in labels:
                valid_features = {
                    key[6:]: value for key, value in features.items()
                    if key[:6] == 'valid#'
                }
                valid_label = labels['valid']
                with tf.variable_scope("model", reuse=True):
                    tf.summary.scalar(
                        'validation-loss',
                        self._model_fn(
                            valid_features, valid_label, mode, params
                        ).loss
                    )
        else:
            with tf.variable_scope("model"):
                spec = self._model_fn(features, labels, mode, params)

        return spec

    def _model_fn(self, features, labels, mode, params):
        raise NotImplementedError('_model_fn should be implemented')

    def train(self, max_steps, dataset=None, valid_dataset=None):
        if self._verbose:
            print('train ...')
        dataset = self._dataset if dataset is None else dataset
        return self._estimator.train(
            input_fn=InputWrapper(dataset, valid_dataset),
            max_steps=max_steps
        )

    def evaluate(self, dataset=None):
        if self._verbose:
            print('evaluate ...')
        dataset = self._dataset if dataset is None else dataset
        return self._estimator.evaluate(input_fn=dataset)

    def predict(self, dataset=None):
        if self._verbose:
            print('predict ...')
        dataset = self._dataset if dataset is None else dataset
        return self._estimator.predict(input_fn=dataset)
