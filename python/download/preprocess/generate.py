
from multiprocessing.pool import Pool
import io
import math
import json
import os.path as path
from zipfile import ZipFile

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from python.download.util.content_dir import ContentDir
from python.operator import make_sequence_example


def char_vocabulary():
    vocabulary_index = np.concatenate([
        np.array([
            '<eos>', '<unknown>', ' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
            'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z'
        ])
    ])

    return dict((v, i) for (i, v) in enumerate(vocabulary_index))


def make_source_target_alignment(words, char_map, max_length,
                                 verbose=False):
    space_char_code = char_map[' ']

    source = []
    source_current = []
    target = []
    target_current = []
    length = []
    length_current = 0

    for word in tqdm(words, disable=not verbose):
        if length_current + len(word) + 1 > max_length:
            # concatenate current data and move it to storage
            source.append(np.concatenate(source_current))
            target.append(np.concatenate(target_current))
            length.append(length_current)

            # prepear for new source and target
            source_current = []
            target_current = []
            length_current = 0

        # add source and target, while maintaining the current total length
        source_current.append(
            np.array([space_char_code] + [char_map[char] for char in word],
                     dtype='int32')
        )
        target_current.append(
            np.array([char_map[char] for char in word] + [space_char_code],
                     dtype='int32')
        )
        length_current += 1 + len(word)

    # concatenate remaning data and move it to storage
    if length_current > 0:
        source.append(np.concatenate(source_current))
        target.append(np.concatenate(target_current))
        length.append(length_current)

    return (length, source, target)


def export_map(vocab_map):
    items_sorted = sorted(vocab_map.items(), key=lambda item: item[1])
    values_sorted = map(lambda item: item[0], items_sorted)

    return np.asarray(list(values_sorted), dtype='str')


def build_dataset(text, max_length=200, verbose=False, **kwargs):
    if verbose:
        print('tokenizing file ...')
    words = np.array(text.strip().split(' '))

    if verbose:
        print('building vocabulary ...')
    char_map = char_vocabulary()

    if verbose:
        print('making dataset ...')
    length, source, target = make_source_target_alignment(
        words, char_map, max_length=max_length,
        verbose=verbose
    )

    return {
        'char_map': export_map(char_map),
        'length': length,
        'source': source,
        'target': target
    }


def split_dataset(dataset, train_ratio=0.9, valid_ratio=0.05, **kwargs):
    # Create indices permutation array
    observations = len(dataset['length'])
    shuffle_indices = np.random.RandomState(2).permutation(observations)

    # Compute number of observations in each dataset
    train_size = math.floor(observations * train_ratio)
    valid_size = math.floor(observations * valid_ratio)

    # Make train split
    train_indices = shuffle_indices[:train_size]
    train = {
        'length': np.take(dataset['length'], train_indices, axis=0),
        'source': np.take(dataset['source'], train_indices, axis=0),
        'target': np.take(dataset['target'], train_indices, axis=0)
    }

    # Make validation split
    valid_indices = shuffle_indices[train_size:train_size + valid_size]
    valid = {
        'length': np.take(dataset['length'], valid_indices, axis=0),
        'source': np.take(dataset['source'], valid_indices, axis=0),
        'target': np.take(dataset['target'], valid_indices, axis=0)
    }

    # Make test split
    test_indices = shuffle_indices[train_size + valid_size:]
    test = {
        'length': np.take(dataset['length'], test_indices, axis=0),
        'source': np.take(dataset['source'], test_indices, axis=0),
        'target': np.take(dataset['target'], test_indices, axis=0)
    }

    return [train, valid, test]


# Serialize dataset in parallel (because it takes forever)
def tfrecord_serializer(item):
    length, source, target = item
    return make_sequence_example(length, source, target).SerializeToString()


def save_tfrecord(filename, dataset, verbose=False):
    observations = len(dataset['length'])

    serialized = []
    with Pool(processes=4) as pool:
        for serialized_string in tqdm(pool.imap(
            tfrecord_serializer,
            zip(dataset['length'], dataset['source'], dataset['target']),
            chunksize=10
        ), total=observations, disable=not verbose):
            serialized.append(serialized_string)

    # Save seriealized dataset
    writer = tf.python_io.TFRecordWriter(
        filename,
        options=tf.python_io.TFRecordOptions(
            tf.python_io.TFRecordCompressionType.ZLIB
        )
    )

    for serialized_string in tqdm(serialized, disable=not verbose):
        writer.write(serialized_string)

    writer.close()


def preprocess_generate(**kwargs):
    with ContentDir() as content:
        content.download('text8.zip', 'http://mattmahoney.net/dc/text8.zip')

    with ZipFile(content.filepath('text8.zip')) as zip_reader:
        with zip_reader.open('text8') as text8_file:
            text = io.TextIOWrapper(text8_file).read()
            dataset = build_dataset(text, **kwargs)
            train, valid, test = split_dataset(dataset, **kwargs)

            print('saving train data ...')
            save_tfrecord(content.filepath('generate.train.tfrecord'),
                          train,
                          verbose=True)

            print('saving valid data ...')
            save_tfrecord(content.filepath('generate.valid.tfrecord'),
                          valid,
                          verbose=True)

            print('saving test data ...')
            save_tfrecord(content.filepath('generate.test.tfrecord'),
                          test,
                          verbose=True)

            print('saving maps ...')
            np.savez(content.filepath('generate.map.npz'),
                     char_map=dataset['char_map'],
                     verbose=True)

            print('saving metadata ...')
            metadata = {
                'observations': {
                    'train': len(train['length']),
                    'valid': len(valid['length']),
                    'test': len(test['length'])
                }
            }
            with open(content.filepath('generate.meta.json'), 'w') as fp:
                json.dump(metadata, fp)
