
import os
import os.path as path

import tensorflow as tf
import pandas as pd

dirname = path.dirname(path.realpath(__file__))
save_dir = path.join(dirname, '..', 'save')
article_data_dir = path.join(dirname, '..', '..', 'article', 'data')


def read_tf_summary(name, alpha=0.25):
    files = os.listdir(path.join(save_dir, name))

    eventfiles = []
    for filename in files:
        if filename[:19] == 'events.out.tfevents':
            eventfiles.append(path.join(save_dir, name, filename))
    eventfiles = sorted(eventfiles)

    rows = []

    prev_wall_time = 0
    file_wall_time_offset = 0

    for eventfile in eventfiles:
        first_event_in_file = True

        for event in tf.train.summary_iterator(eventfile):
            if first_event_in_file:
                file_wall_time_offset = prev_wall_time - event.wall_time
                first_event_in_file = False

            for value in event.summary.value:
                if value.tag == 'loss':
                    rows.append({
                        'dataset': 'train',
                        'sec': event.wall_time + file_wall_time_offset,
                        'loss': value.simple_value,
                        'step': event.step
                    })
                elif value.tag == 'model_1/validation-loss':
                    rows.append({
                        'dataset': 'valid',
                        'sec': event.wall_time + file_wall_time_offset,
                        'loss': value.simple_value,
                        'step': event.step
                    })

        prev_wall_time = event.wall_time + file_wall_time_offset

    df = pd.DataFrame(rows, columns=('dataset', 'sec', 'loss', 'step'))
    df.set_index(['dataset', 'sec', 'step'], inplace=True)
    df['loss smooth'] = df['loss'].ewm(alpha=alpha).mean()

    return df


autocomplete = pd.concat(
    [
        read_tf_summary('autocomplete_gru_600'),
        read_tf_summary('autocomplete_lstm_600'),
        read_tf_summary('autocomplete_nlstm_600')
    ],
    keys=['GRU', 'LSTM', 'Nested LSTM'],
    names=['model']
)
autocomplete.to_csv(path.join(article_data_dir, 'autocomplete-training.csv'))

generate = pd.concat(
    [
        read_tf_summary('generate_gru_1200'),
        read_tf_summary('generate_lstm_1200'),
        read_tf_summary('generate_nlstm_1200')
    ],
    keys=['GRU', 'LSTM', 'Nested LSTM'],
    names=['model']
)
generate.to_csv(path.join(article_data_dir, 'generate-training.csv'))
