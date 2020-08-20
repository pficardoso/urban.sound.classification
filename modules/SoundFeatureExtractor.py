import librosa
import numpy as np


class SoundFeatureExtractor():

    def __init__(self):
        pass

    def load_audio_file(self, audio_file_path, sr=None):
        self.y, sr = librosa.load(audio_file_path, sr)
        self.sr = sr

    def extract_all_ft(self, audio_file_path=None):

        if audio_file_path:
            self.load_audio_file(audio_file_path)
        else:
            ## check if there is audiofile
            if not self.y:
                raise Exception("Audio file path was not loaded yet.")

        return {"zcr"       : self.extract_zero_cross_rate_ft(),
                "freq_centr": self.extract_spectral_centroid_ft(),
                "spec_rol"  : self.extract_spectral_rolloff_ft(),
                "mfccs"     : self.extract_mfcc_ft()}

    def extract_zero_cross_rate_ft(self):
        zcr = librosa.feature.zero_crossing_rate(self.y)[0]
        return {"avg": zcr.mean(), "sum": zcr.sum()}

    def extract_spectral_centroid_ft(self):
        ctr = librosa.feature.spectral_centroid(self.y)[0]
        return {"avg": ctr.mean(), "sum": ctr.sum()}

    def extract_spectral_rolloff_ft(self, roll_percent=0.85):
        rol = librosa.feature.spectral_rolloff(self.y, self.sr, roll_percent=roll_percent)[0]
        return {"avg":rol.mean(), "sum":rol.sum()}

    def extract_mfcc_ft(self, n_mfcc=20):
        mfccs = librosa.feature.mfcc(self.y, self.sr, n_mfcc=n_mfcc)
        ## mfccs is a taple of size (n_mfccs, nr_frames)
        result_avg = np.zeros(len(mfccs))
        result_sum = np.zeros(len(mfccs))
        for mfcc_i in np.arange(len(mfccs)):
            result_avg[mfcc_i] = mfccs[mfcc_i].mean()
            result_sum[mfcc_i] = mfccs[mfcc_i].sum()

        return {"avg":result_avg, "sum":result_sum}
    


