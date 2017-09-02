all: hls

model:
	python model.py

hls: model
	cp -r data ../hw/hls/auto_enc/.

clean:
	rm -rf data
