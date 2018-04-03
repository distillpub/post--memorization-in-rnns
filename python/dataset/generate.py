
import json

import tensorflow as tf
import numpy as np

from python.download import load_content
from python.operator import parse_sequence_example


class Generate:
    def __init__(self,
                 observations: int=None, batch_size: int=128,
                 dataset: str='train', repeat: bool=False):
        # save config
        self._observations = observations
        self._batch_size = batch_size
        self._repeat = repeat

        # load data
        content = load_content('generate')
        maps = np.load(content['map.npz'])

        # Store maps for later, note these are inverse of those used in
        # the preprocessing step.
        self.char_map = maps['char_map']

        # Get number of classes
        self.source_classes = len(self.char_map)
        self.target_classes = len(self.char_map)

        # select either test or train dataset
        if dataset == 'train':
            self._tfrecord = content['train.tfrecord']
        elif dataset == 'valid':
            self._tfrecord = content['valid.tfrecord']
        elif dataset == 'test':
            self._tfrecord = content['test.tfrecord']
        else:
            raise ValueError(f'the dataset type {dataset} is invalid')

        # Compute dataset statistics
        with open(content['meta.json']) as fp:
            meta = json.load(fp)
            self.observations = meta['observations'][dataset]
            self.batches = self.observations // batch_size

    def decode_source(self, classes):
        return self.char_map[classes]

    def decode_target(self, classes):
        return self.char_map[classes]

    def _parse_example(self, example_proto):
        length, source, target = parse_sequence_example(example_proto)

        # convert type
        length = tf.cast(length, dtype=tf.int32)
        source = tf.cast(source, dtype=tf.int32)
        target = tf.cast(target, dtype=tf.int32)

        return (length, source, target)

    def __call__(self):
        # Create shuffled batches
        # Documentation: https://www.tensorflow.org/programmers_guide/datasets
        dataset = tf.data.TFRecordDataset([self._tfrecord],
                                          compression_type="ZLIB")
        if self._observations is not None:
            dataset = dataset.take(self._observations)
        dataset = dataset.map(self._parse_example)
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.padded_batch(self._batch_size, (
            [],
            [None],
            [None]
        ))
        if self._repeat:
            dataset = dataset.repeat()

        # Export iterator
        iterator = dataset.make_one_shot_iterator()
        length, source, target = iterator.get_next()

        features = {
            'source': source,
            'target': target,
            'length': length
        }
        return (features, target)
