all: model

./data/data_i.out:
	mkdir -p data
	python model.py

hw: ./data/data_i.out
	cp -r data ../hw/hls/auto_enc/.
	cp data/*.vh ../hw/testbenches/noc_block_autoenc_tb/.


clean:
	rm -rf data __pycache__
