
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from python.dataset import AutoComplete
from python.model import PureNLSTM
from python.summary import ContextAccuracy

tf.logging.set_verbosity(tf.logging.INFO)

train_dataset = AutoComplete(repeat=True, batch_size=64)
test_dataset = AutoComplete(dataset='test', repeat=False, batch_size=16)
model = PureNLSTM(train_dataset, name='autocomplete_nlstm_600',
                  embedding_size=600,
                  verbose=True)

context_accuracy = ContextAccuracy()

for output_i, output in enumerate(
    tqdm(model.predict(dataset=test_dataset), total=test_dataset.observations)
):
    probabilities = output['probabilities']
    predict_sorted = np.argsort(probabilities, axis=1)[:, ::-1]

    source = test_dataset.decode_source(output['source'])
    target = test_dataset.decode_target(output['target'])
    predict = test_dataset.decode_target(predict_sorted)

    context_accuracy.add(source, predict, target)

print(context_accuracy.summary())
