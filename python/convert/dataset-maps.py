
import sys
import json

from python.dataset.auto_complete import AutoComplete

if sys.argv[1] == 'autocomplete':
    dataset = AutoComplete()
else:
    raise NotImplementedError(f'{sys.argv[1]} is an unknown dataset')

maps = {
    'char_map': list(dataset.char_map),
    'word_map': list(dataset.word_map)
}

json.dump(maps, sys.stdout, indent=2)
