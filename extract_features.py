import argparse
from modules.SoundsDatabase import SoundsDatabase
from modules.SoundFeatureExtractor import SoundFeatureExtractor
from tqdm import tqdm
import pickle

## open file with words input
parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", metavar="OUTPUT_PATH" , help="Path for the output")

if __name__=="__main__":

    args = parser.parse_args()
    print(args.output)
    soundsDB = SoundsDatabase("./data/train.csv", "./data/audio.files")
    soundsFtExtractor = SoundFeatureExtractor()

    print("Extracting features from soundsDB")
    ids = list()
    data = list()
    labels = list()

    ## Extract all features
    for audiofile in tqdm(soundsDB):
        data.append(soundsFtExtractor.extract_all_ft(audiofile))


    output_object = {"ids": soundsDB.get_ids(), "data": data, "labels": soundsDB.get_labels()}

    pickle.dump( output_object, open( "".join([args.output,".pickle"]) , "wb" ) )

