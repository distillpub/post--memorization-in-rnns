
import json
import numpy as np
import os.path as path
import tensorflow as tf

from python.dataset import AutoComplete, AutoCompleteFixed
from python.model import PureNLSTM

dirname = path.dirname(path.realpath(__file__))
article_dir = path.join(dirname, '..', '..', 'article')

tf.logging.set_verbosity(tf.logging.INFO)

train_dataset = AutoComplete(repeat=True)
test_dataset = AutoCompleteFixed(
    "context the formal study of grammar is an important part of education "
    "from a young age through advanced learning though the rules taught in "
    "schools are not a grammar in the sense most linguists use",
    batch_size=1,
    use_offsets=True
)
model = PureNLSTM(train_dataset, name='autocomplete_nlstm_600',
                  embedding_size=600,
                  verbose=True)

data = []

print(f'sequence:')
for output_i, output in enumerate(
    model.predict(dataset=test_dataset)
):
    probabilities = output['probabilities']
    predict_sorted = np.argsort(probabilities, axis=1)[:, ::-1]

    source = test_dataset.decode_source(output['source'])
    target = test_dataset.decode_target(output['target'])
    predict = test_dataset.decode_target(predict_sorted)
    connectivity = output['connectivity']

    char = source[output_i]
    words_sorted = predict[output_i]
    target_word = target[output_i]

    print(f' {char} -> {words_sorted[0]}, {words_sorted[1]},'
          f' {words_sorted[2]}'
          f' -- {target_word}')

    data.append({
        'char': char,
        'target': target_word,
        'predict': words_sorted[:5].tolist(),
        'connectivity': connectivity.tolist()
    })

with open(path.join(article_dir, 'data/connectivity_nlstm.json'), 'w') as file:
    json.dump(data, file)

print('done')
