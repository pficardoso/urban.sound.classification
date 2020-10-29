Use conda env "urban.classification.sound"


## 1 - Extraction of features

Run: 
python extract_features.py -o pickle.repo/data.features.v1.pickle

## 2 - Use script to indentify class of an audio 

Run:

python3 test.urban.sounds.classification.py -i <wav.file path> -m <model.path> -l <label.enconder.pickle path>

Ex:
python3 test.urban.sounds.classification.py -i data/audio.files/2010.wav -m models.repo/model.fully.connected.network.h5 -l pickle.repo/model.fully.connected.network.label.encoder.pickle
