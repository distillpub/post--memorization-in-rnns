
import re
import os
import json
import os.path as path

import yaml
import tensorflow as tf

dirname = path.dirname(path.realpath(__file__))
save_dir = path.join(dirname, '..', 'save')
article_save_dir = path.join(dirname, '..', '..', 'article', 'save')


class ExportCheckpoint:
    def __init__(self, checkpoint_name, verbose=False):
        self._verbose = verbose
        self._checkpoint_name = checkpoint_name

        # Create reader
        with open(path.join(save_dir, checkpoint_name, 'checkpoint')) as fp:
            latest_checkpoint = yaml.load(fp)['model_checkpoint_path']
            self._reader = tf.train.NewCheckpointReader(
                path.join(save_dir, checkpoint_name, latest_checkpoint)
            )

        # Build manifest
        self._manifest = {}

        variables = self._reader.get_variable_to_shape_map()
        for (name, shape) in variables.items():
            if (
                name == 'global_step' or
                re.match('^.+/Adam(_1)?(/|$)', name) or
                re.match('^.+/beta(1|2)_power$', name)
            ):
                continue

            self._manifest[name] = {
                'filename': self.make_filename(name),
                'shape': shape
            }

    def make_filename(self, name):
        name = name.replace('/', '_')
        name = re.sub("[^A-Z0-9_]", "", name, flags=re.IGNORECASE)
        return name

    def export_to(self, export_dir):
        if self._verbose:
            print(f'exporting {self._checkpoint_name} checkpoint')
        # Make directory
        os.makedirs(export_dir, exist_ok=True)

        # Create manifest
        if self._verbose:
            print(f'  exporting manifest: manifest.json')
        with open(path.join(export_dir, 'manifest.json'), 'w') as fp:
            json.dump(self._manifest, fp, indent=2, sort_keys=True)

        # Dump weights
        for name, info in self._manifest.items():
            if self._verbose:
                print(f'  exporting tensor: {name}')
            tensor = self._reader.get_tensor(name)
            with open(path.join(export_dir, info['filename']), 'wb') as fp:
                fp.write(tensor.tobytes())


ExportCheckpoint('autocomplete_gru_600', verbose=True).export_to(
    path.join(article_save_dir, 'autocomplete_gru')
)
ExportCheckpoint('autocomplete_lstm_600', verbose=True).export_to(
    path.join(article_save_dir, 'autocomplete_lstm')
)
ExportCheckpoint('autocomplete_nlstm_600', verbose=True).export_to(
    path.join(article_save_dir, 'autocomplete_nlstm')
)

ExportCheckpoint('generate_gru_1200', verbose=True).export_to(
    path.join(article_save_dir, 'generate_gru')
)
ExportCheckpoint('generate_lstm_1200', verbose=True).export_to(
    path.join(article_save_dir, 'generate_lstm')
)
ExportCheckpoint('generate_nlstm_1200', verbose=True).export_to(
    path.join(article_save_dir, 'generate_nlstm')
)
