all: model

model:
	mkdir -p data
	python model_fft.py

hw: model
	cp -r data ../hw/hls/auto_enc/.
	cp data/*.vh ../hw/testbenches/noc_block_autoenc_tb/.

clean:
	rm -rf data
