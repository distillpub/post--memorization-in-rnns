
import numpy as np


class ContextAccuracy:
    def __init__(self, predictions=3, max_word_length=50):
        self._predictions = predictions
        self._longest_word = 0

        self._accuracy_count = np.zeros(
            (predictions, max_word_length + 1),
            dtype=np.int32)
        self._accuracy_total = np.zeros(
            (max_word_length + 1, ),
            dtype=np.int32)

    def add(self, source, predict, target):
        word_chars_consumed = 0
        for char, predict_sorted, target_word in zip(source, predict, target):
            if char == ' ':
                word_chars_consumed = 0
            else:
                word_chars_consumed += 1

            self._longest_word = max(self._longest_word, word_chars_consumed)
            self._accuracy_total[word_chars_consumed] += 1

            for prediction_i in range(self._predictions):
                if predict_sorted[prediction_i] == target_word:
                    self._accuracy_count[
                        prediction_i:, word_chars_consumed] += 1

    def summary(self):
        return (self._accuracy_count[:, :(self._longest_word + 1)] /
                self._accuracy_total[np.newaxis, :(self._longest_word + 1)])
