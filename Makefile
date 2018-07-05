
.PHONEY: train bundle convert server convert sync fetch-python fetch-article

train:
	PYTHONPATH=./ python3 python/run/autocomplete_pure_gru_train.py
	PYTHONPATH=./ python3 python/run/autocomplete_pure_lstm_train.py
	PYTHONPATH=./ python3 python/run/autocomplete_pure_nlstm_train.py
	PYTHONPATH=./ python3 python/run/generate_pure_gru_train.py
	PYTHONPATH=./ python3 python/run/generate_pure_lstm_train.py
	PYTHONPATH=./ python3 python/run/generate_pure_nlstm_train.py

bundle:
	browserify article/main.js --debug > article/bundle.js
	browserify article/feedback-main.js --debug > article/feedback-bundle.js
	browserify article/feedback-3-main.js --debug > article/feedback-3-bundle.js

bundle-watch:
	watchify article/main.js --debug -o article/bundle.js

bundle-production:
	browserify article/main.js -t babelify > article/bundle.js
	browserify article/feedback-main.js -t babelify > article/feedback-bundle.js
	browserify article/feedback-3-main.js -t babelify > article/feedback-3-bundle.js

server:
	http-server article

convert:
	PYTHONPATH=./ python3 python/convert/export-checkpoint.py
	PYTHONPATH=./ python3 python/convert/dataset-maps.py autocomplete > article/data/autocomplete.json
	PYTHONPATH=./ python3 python/convert/precompute.py
	PYTHONPATH=./ python3 python/convert/tfsummary.py

sync:
	rsync --info=progress2 -urltv --delete \
		--exclude 'python/save-old' --exclude 'python/save' --exclude 'python/download/content' \
		--exclude 'article/save' --exclude 'article/node_modules' \
		--exclude 'article/bundle.js' --exclude 'article/data' \
		-e ssh ./ nearform-gpu:~/workspace/01-nested-lstm

fetch-python:
	rsync --info=progress2 -urltv --delete \
	-e ssh nearform-gpu:~/workspace/01-nested-lstm/python/save/ ./python/save
	rsync --info=progress2 -urltv --delete \
	-e ssh nearform-gpu:~/workspace/01-nested-lstm/python/download/content/ ./python/download/content

fetch-article:
	rsync --info=progress2 -urltv --delete \
	-e ssh nearform-gpu:~/workspace/01-nested-lstm/article/save/ ./article/save
	rsync --info=progress2 -urltv --delete \
	-e ssh nearform-gpu:~/workspace/01-nested-lstm/article/data/ ./article/data
