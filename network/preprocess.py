"""
for each file in dataset:
1- load file
2- pad the signal (if necessary)
3- extracting log spectrogram from signal (using librosa)
4- normalise spectrogram
5- save the normalised spectrogram

PreprocessingPipeline() class
"""
import os
import pickle

import librosa
import numpy as np


class Loader:
    """Loader is responsible for loading an audio file.
    (librosa wrapper)"""

    def __init__(self, sample_rate, duration, mono):
        self.sample_rate = sample_rate
        self.duration = duration
        self.mono = mono

    def load(self, file_path):
        signal = librosa.load(file_path,
                              sr=self.sample_rate,
                              duration=self.duration,
                              mono=self.mono)[0] # first index is signal
        return signal


class Padder:
    """Padder is responsible for applying padding to an array.
    (numpy wrapper)"""
    
    def __init__(self, mode="constant"): # mode is type of padding to apply
        self.mode = mode


    # let's say we have array [1,2,3]
    # want to pad it with 2 items
    # there are 2 ways:
    # 1) prepend: [0,0,1,2,3] = left (zero)padding
    # 2) append: [1,2,3,0,0] = right (zero)padding
    # mode=constant fills values with a constant (here, zero)
    def left_pad(self, array, num_missing_items):
        padded_array = np.pad(array,
                              (num_missing_items, 0),
                              mode=self.mode)
        return padded_array

    def right_pad(self, array, num_missing_items):
        padded_array = np.pad(array,
                              (0,num_missing_items),
                              mode=self.mode)
        return padded_array


class LogSpectrogramExtractor:
    """LogSpectrogramExtractor extracts log spectrograms (in dB) from a
    time-series signal."""
    
    def __init__(self, frame_size, hop_length):
        self.frame_size = frame_size
        self.hop_length = hop_length

    def extract(self, signal):
        # extract short-time fourier transform
        stft = librosa.stft(signal,
                            n_fft=self.frame_size,
                            hop_length=self.hop_length)[:-1]
        # returns 2D array with shape: (1 + frame_size / 2, num_frames)
        # typcial frame size: 1024 -> 513 but we want even numbers
        # so drop final frequency bin, to remain with 512.
        spectrogram = np.abs(stft)
        log_spectrogram = librosa.amplitude_to_db(spectrogram)
        return log_spectrogram

class MinMaxNormaliser:
    """MinMaxNormaliser applies min max normalisation to an array.
    (Take array with different values, min value mapped to 0, max value mapped to 1)"""
    
    def __init__(self, min_val, max_val):
        self.min = min_val
        self.max = max_val

    def normalise(self, array):
        # print("########################1")
        # print(array)
        # print(type(array))
        # print("########################1")
        norm_array = (array - array.min()) / (array.max() - array.min()) # makes max = 1, min = 0
        norm_array = norm_array * (self.max - self.min) + self.min # makes max = self.max, min = self.min
        return norm_array

    def denormalise(self, norm_array, original_min, original_max):
        array = (norm_array - self.min) / (self.max - self.max) # invert line 86
        array = array * (original_max - original_min) + original_min # invert line 85
        return array

class Saver:
    """Saver is responsible for saving features, and the min and max values"""
    
    def __init__(self, feature_save_dir, min_max_values_save_dir):
        self.feature_save_dir = feature_save_dir
        self.min_max_values_save_dir = min_max_values_save_dir

    def save_feature(self, feature, file_path):
        save_path = self._generate_save_path(file_path)
        np.save(save_path, feature)

    def save_min_max_values(self, min_max_values):
        save_path = os.path.join(self.min_max_values_save_dir,
                                 "min_max_values.pkl")
        self._save(min_max_values, save_path)

    @staticmethod # because not using any attributes or methods from this class
    def _save(data, save_path):
        with open(save_path, "wb") as f:
            pickle.dump(data, f)

    def _generate_save_path(self, file_path):
        file_name = os.path.split(file_path)[1] # split returns head, tail (we want tail)
        save_path = os.path.join(self.feature_save_dir, file_name + ".npy")
        return save_path

class PreprocessingPipeline:
    """PreprocessingPipeline processes audio files in a directory, applying
    the following stteps to each file:
        1- load file
        2- pad the signal (if necessary)
        3- extracting log spectrogram from signal (using librosa)
        4- normalise spectrogram
        5- save the normalised spectrogram


    Storing the min max values for all the log spectrograms (for reconstructing
    the signal, denormalising it).
    """
    
    def __init__(self):
        # Need an instance of all above classes above in PreprocessingPipeline
        self.padder = None
        self.extractor = None
        self.normaliser = None
        self.saver = None
        self.min_max_values = {} # save path: min, max
        self._loader = None
        self._num_expected_samples = None

    @property
    def loader(self):
        return self._loader
    
    # every time we set the loader, we're going to set number of expected samples
    @loader.setter
    def loader(self, loader):
        self._loader = loader
        self._num_expected_samples = num_expected_samples = int(loader.sample_rate * loader.duration)

    # main method exposed to client code
    def process(self, audio_files_dir):
        # loop through audio files in dir
        for root, _, files in os.walk(audio_files_dir):
            for file in files:
                file_path = os.path.join(root, file)
                self._process_file(file_path)
                print(f"Processed file {file_path}")
        self.saver.save_min_max_values(self.min_max_values)

    def _process_file(self, file_path):
        signal = self.loader.load(file_path)
        # print("########################3")
        # print(signal)
        # print(type(signal))
        # print("########################3")
        # print("########################4")
        # print(self._is_padding_necessary(signal))
        # print(type(self._is_padding_necessary(signal)))
        # print("########################4")       

        if self._is_padding_necessary(signal):
            signal = self._apply_padding(signal)
        feature = self.extractor.extract(signal) # getting log spectrogram here
        # print("########################2")
        # print(feature)
        # print(type(feature))
        # print("########################2")
        norm_feature = self.normaliser.normalise(feature)
        save_path = self.saver.save_feature(norm_feature, file_path)
        self._store_min_max_value(save_path, feature.min(), feature.max())

    def _is_padding_necessary(self, signal):
        """given we load a certain duration, and duration is fixed,
        and we know the sample rate, we know the number of expected samples"""
        # if signal has fewer samples than expected, audio file is shorter -> pad
        if len(signal) < self._num_expected_samples:
            return True
        return False
    
    def _apply_padding(self, signal):
        num_missing_samples = self._num_expected_samples - len(signal)
        padded_signal = self.padder.right_pad(signal, num_missing_samples)
        return padded_signal
    
    def _store_min_max_value(self, save_path, min_val, max_val):
        self.min_max_values[save_path] = {
            "min": min_val,
            "max": max_val
        }

if __name__ == "__main__":
    FRAME_SIZE = 512
    HOP_LENGTH = 256
    DURATION = 0.74 # in seconds (gives a nice number of frames)
    SAMPLE_RATE = 22050
    MONO = True

    current_directory = os.getcwd()
    path_head = os.path.split(current_directory)[0]

    SPECTROGRAMS_SAVE_DIR = os.path.join(path_head, "datasets/fsdd/spectrograms/")
    MIN_MAX_VALUES_SAVE_DIR = os.path.join(path_head, "datasets/fsdd/")
    FILES_DIR = os.path.join(path_head, "datasets/fsdd/audio/")

    # instanciate all objects
    loader = Loader(SAMPLE_RATE, DURATION, MONO)
    padder = Padder()
    log_spectrogram_extractor = LogSpectrogramExtractor(FRAME_SIZE, HOP_LENGTH)
    min_max_normaliser = MinMaxNormaliser(0, 1)
    saver = Saver(SPECTROGRAMS_SAVE_DIR, MIN_MAX_VALUES_SAVE_DIR)

    preprocessing_pipeline = PreprocessingPipeline()
    preprocessing_pipeline.loader = loader
    preprocessing_pipeline.padder = padder
    preprocessing_pipeline.extractor = log_spectrogram_extractor
    preprocessing_pipeline.normaliser = min_max_normaliser
    preprocessing_pipeline.saver = saver

    preprocessing_pipeline.process(FILES_DIR)