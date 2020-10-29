import argparse
from modules.SoundFeatureExtractor import SoundFeatureExtractor
import tensorflow.keras as K
import pickle
import os

## Removes some unuseful warnings of tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

### configure command line
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_audio", metavar="PATH",
                    help="Path to audio file to classify")
parser.add_argument("-m","--model", metavar="PATH",
                    help="Path to model")
parser.add_argument("-l","--label_encoder",
                    metavar="PATH", help="Path to label encoder")


if __name__ == '__main__':

    args = parser.parse_args()
    audio_path = args.input_audio
    model_path = args.model
    label_encoder_path = args.label_encoder

    print("Extracting features")
    ### load of audio to test
    sound_extractor = SoundFeatureExtractor()
    audio_features = sound_extractor.extract_all_ft(audio_path)
    flat_audio_features = sound_extractor.flat_all_features(audio_features)

    print("Loading Model")
    ### load of model
    model = K.models.load_model(model_path)

    print("Loading Label Encoder")
    labelEncoder = pickle.load( open( label_encoder_path, "rb" ))

    print("Predicting")
    ### run
    model_output = model(flat_audio_features)
    model_output_one_hot = labelEncoder.inverse_transform(model_output.numpy())

    ### print result
    print("Result:", model_output_one_hot)
