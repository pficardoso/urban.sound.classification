import pandas as pd
from os import path
from modules.SoundFeatureExtractor import  SoundFeatureExtractor


class SoundsDatabase():

    def __init__(self, csv_path, audiofiles_dir_path):
        self.csv_path  = path.abspath(csv_path)
        self.audiofiles_dir = path.abspath(audiofiles_dir_path)
        self.data_df = pd.read_csv(csv_path).set_index("ID")


    def __len__(self):
        return self.data_df.shape[0]

    def __getitem__(self, item):
        audio_name = "".join([str(self.data_df.iloc[item].name), ".wav"])
        return "/".join([self.audiofiles_dir, audio_name])



soundsDB = SoundsDatabase("./data/train.csv", "./data/audio.files")
soundsFtExtractor = SoundFeatureExtractor()

for file in soundsDB:
    print(file)

