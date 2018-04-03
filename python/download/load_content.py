
import os.path as path

import numpy as np

from python.download.util.content_dir import ContentDir
from python.download.preprocess.autocomplete import preprocess_autocomplete
from python.download.preprocess.generate import preprocess_generate


def load_content(name: str):
    with ContentDir() as content:
        paths = {
            'train.tfrecord': content.filepath(name + '.train.tfrecord'),
            'valid.tfrecord': content.filepath(name + '.valid.tfrecord'),
            'test.tfrecord': content.filepath(name + '.test.tfrecord'),
            'meta.json': content.filepath(name + '.meta.json'),
            'map.npz': content.filepath(name + '.map.npz')
        }

        if name == 'autocomplete':
            if not content.exists(name + '.train.tfrecord'):
                preprocess_autocomplete(
                    train_ratio=0.9, valid_ratio=0.05,
                    vocab_size=2**14, max_length=200,
                    verbose=True
                )
        elif name == 'generate':
            if not content.exists(name + '.train.tfrecord'):
                preprocess_generate(
                    train_ratio=0.9, valid_ratio=0.05,
                    max_length=180, verbose=True
                )
        else:
            raise NotImplementedError(
                f'the dataset "{name}" is not implemented'
            )

    return paths
