
import tensorflow as tf
import numpy as np

from python.download import load_content
from python.operator import parse_sequence_example


class AutoCompleteFixed:
    def __init__(self, sample: str,
                 observations: int=None, batch_size: int=128,
                 use_offsets=False,
                 repeat: bool=False):
        # save config
        self._observations = observations
        self._batch_size = batch_size
        self._repeat = repeat
        self._use_offsets = use_offsets

        # load data
        content = load_content('autocomplete')
        maps = np.load(content['map.npz'])

        # Store maps for later, note these are inverse of those used in
        # the preprocessing step.
        self.char_map = maps['char_map']
        self.word_map = maps['word_map']

        # Create inverse maps
        self._char_map_inverse = {c: i for (i, c) in enumerate(self.char_map)}
        self._word_map_inverse = {w: i for (i, w) in enumerate(self.word_map)}

        # Get number of classes
        self.source_classes = len(self.char_map)
        self.target_classes = len(self.word_map)

        # Encode sample
        length, source, target = self.make_source_target_alignment(sample)
        if self._use_offsets:
            lengths = np.tile(length, (length))
            sources = np.tile(source, (length, 1))
            targets = np.tile(target, (length, 1))
            offsets = np.arange(length, dtype='int32')
        else:
            lengths = np.asarray([length], dtype='int32')
            sources = source[np.newaxis, ...]
            targets = target[np.newaxis, ...]
            offsets = np.asarray([0], dtype='int32')

        self._records = (
           lengths, sources, targets, offsets
        )

    def make_source_target_alignment(self, sequence):
        space_char_code = self._char_map_inverse[' ']
        unknown_word_code = self._word_map_inverse['<unknown>']

        source = []
        target = []
        length = 0

        for word in sequence.split(' '):
            source.append(
                np.array([space_char_code] + self.encode_source(word),
                         dtype='int32')
            )
            target.append(
                np.full(len(word) + 1, self.encode_target([word])[0],
                        dtype='int32')
            )
            length += 1 + len(word)

        # concatenate data
        return (
            length,
            np.concatenate(source),
            np.concatenate(target)
        )

    def encode_source(self, chars):
        return [self._char_map_inverse[char] for char in chars]

    def encode_target(self, words):
        for word in words:
            if word not in self._word_map_inverse:
                raise ValueError(f'the word "{word}" is not in the vocabulary')

        return [self._word_map_inverse[word] for word in words]

    def decode_source(self, classes):
        return self.char_map[classes]

    def decode_target(self, classes):
        return self.word_map[classes]

    def __call__(self):
        # Create shuffled batches
        # Documentation: https://www.tensorflow.org/programmers_guide/datasets
        dataset = tf.data.Dataset.from_tensor_slices(self._records)
        if self._observations is not None:
            dataset = dataset.take(self._observations)
        if self._repeat:
            dataset = dataset.repeat()
        dataset = dataset.batch(self._batch_size)

        # Export iterator
        iterator = dataset.make_one_shot_iterator()
        length, source, target, offset = iterator.get_next()

        features = {
            'source': source,
            'target': target,
            'length': length
        }
        if (self._use_offsets):
            features['offset'] = offset

        return (features, target)
