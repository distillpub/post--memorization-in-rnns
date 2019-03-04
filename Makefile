
.PHONEY: train build convert bundle bundle-watch bundle-production server convert sync fetch-python fetch-article

train:
	PYTHONPATH=./ python3 python/run/autocomplete_pure_gru_train.py
	PYTHONPATH=./ python3 python/run/autocomplete_pure_lstm_train.py
	PYTHONPATH=./ python3 python/run/autocomplete_pure_nlstm_train.py
	PYTHONPATH=./ python3 python/run/generate_pure_gru_train.py
	PYTHONPATH=./ python3 python/run/generate_pure_lstm_train.py
	PYTHONPATH=./ python3 python/run/generate_pure_nlstm_train.py

build: quantitative convert connectivity

quantitative:
	PYTHONPATH=./ python3 python/run/autocomplete_pure_gru_quantitative.py
	PYTHONPATH=./ python3 python/run/autocomplete_pure_lstm_quantitative.py
	PYTHONPATH=./ python3 python/run/autocomplete_pure_nlstm_quantitative.py

convert:
	PYTHONPATH=./ python3 python/convert/export-checkpoint.py
	PYTHONPATH=./ python3 python/convert/dataset-maps.py autocomplete > public/data/autocomplete.json
	PYTHONPATH=./ python3 python/convert/precompute.py
	PYTHONPATH=./ python3 python/convert/tfsummary.py

connectivity:
	PYTHONPATH=./ python3 python/run/autocomplete_pure_gru_connectivity.py
	PYTHONPATH=./ python3 python/run/autocomplete_pure_lstm_connectivity.py
	PYTHONPATH=./ python3 python/run/autocomplete_pure_nlstm_connectivity.py

evaluate-autocomplete:
	PYTHONPATH=./ python3 python/run/autocomplete_pure_gru_test.py
	PYTHONPATH=./ python3 python/run/autocomplete_pure_lstm_test.py
	PYTHONPATH=./ python3 python/run/autocomplete_pure_nlstm_test.py

evaluate-generate:
	PYTHONPATH=./ python3 python/run/generate_pure_gru_test.py
	PYTHONPATH=./ python3 python/run/generate_pure_lstm_test.py
	PYTHONPATH=./ python3 python/run/generate_pure_nlstm_test.py

bundle:
	npm run bundle

bundle-watch:
	npm run bundle-watch

bundle-production:
	npm run bundle-production

server:
	npm run start

sync:
	rsync --info=progress2 -urltv --delete \
		--exclude 'python/save-old' --exclude 'python/save' --exclude 'python/download/content' \
		--exclude 'public/save' --exclude 'public/node_modules' \
		--exclude 'public/bundle.js' --exclude 'public/data' \
		-e ssh ./ amazon-gpu:~/workspace/connectivity

fetch-python:
	rsync --info=progress2 -urltv --delete \
	-e ssh amazon-gpu:~/workspace/connectivity/python/save/ ./python/save
	rsync --info=progress2 -urltv --delete \
	-e ssh amazon-gpu:~/workspace/connectivity/python/download/content/ ./python/download/content

fetch-article:
	rsync --info=progress2 -urltv --delete \
	-e ssh amazon-gpu:~/workspace/connectivity/article/save/ ./public/save
	rsync --info=progress2 -urltv --delete \
	-e ssh amazon-gpu:~/workspace/connectivity/article/data/ ./public/data
