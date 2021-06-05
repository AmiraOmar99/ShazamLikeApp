import os
import io 
import json
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import librosa.display
from librosa.core import load
from pydub import AudioSegment
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import (generate_binary_structure, iterate_structure, binary_erosion)
import logging
import imagehash
from PIL import Image
# import difflib
from imagehash import hex_to_hash

logger = logging.getLogger()

curr_dir = os.getcwd()

class Song():
    def __init__(self,path= ""):
        self.path=path
        self.data = None
        self.sr = None
        self.hop_length = 512 
        self.window_size = 2048
        # self.mel_spectrogram =None
        # self.mfcc = None
        # self.chroma_stft = None
        # self.o_env=None
        # self.onset_frames= None
        # self.spectral_centroids=None
        self.features={"mel_spectrogram": None, "mfcc":None,"chroma_stft":None, "spectral_contrast":None}
        self.hashed_features= {"mel_spectrogram": None, "mfcc":None,"chroma_stft":None,"spectral_contrast":None }
        self.read_song()
        
    def read_song(self):
        self.name, ext = os.path.splitext(self.path)
        
        if ext in [".mp3",".MP3"]:
            sound = AudioSegment.from_mp3(self.path)
            sound.export("converted.wav", format="wav")
            path = "./converted.wav"

        self.data, self.sr = librosa.core.load(self.path, mono=True, duration=60)

    def gen_spectrogram(self):
        self.features["mel_spectrogram"] = librosa.feature.melspectrogram(self.data, sr=self.sr, n_fft =self.window_size , hop_length=self.hop_length)
        self.features["mel_spectrogram"] = librosa.power_to_db(self.features["mel_spectrogram"])

    def save_spectrogram(self,path):
        fig = plt.Figure()       
        ax = fig.add_subplot(111)
        librosa.display.specshow(self.features["mel_spectrogram"], sr=self.sr, hop_length=self.hop_length, x_axis='time', y_axis='mel')
        plt.savefig(path+os.path.split(self.name)[1])
 
    def get_features(self):
        self.features["mfcc"] = librosa.feature.mfcc(y=self.data.astype('float64'),  n_mfcc=20, sr=self.sr).tolist()
        self.features["chroma_stft"] =librosa.feature.chroma_stft(y= self.data, sr=self.sr).tolist()
        self.features["spectral_contrast"]= librosa.feature.spectral_contrast(y=self.data, sr=self.sr).tolist()
        self.features["mel_spectrogram"] =self.features["mel_spectrogram"].tolist()

    def createPerceptualHash(self, feature) -> str:
        logger.debug("Creating Perceptual Hash for each feature")
        dataInstance = Image.fromarray(feature)
        hashed_data = imagehash.phash(dataInstance, hash_size=16).__str__()
        # print(type(hashed_data))
        return hashed_data

    def getHashedData(self, Hdic , Fdic):
        for key in Hdic:
            Hdic[key]=self.createPerceptualHash(np.array(Fdic[key]))
    
    def write_json(self, PATH, dict):
        with io.open(PATH, 'w') as db_file:
            json.dump(dict,db_file)

    # # gets the similarity index between this song and another song features
    # def get_similarity_index(self, compared_features):
    #     sim_index = []
    #     for hash in compared_features:
    #         sim_index.append(difflib.SequenceMatcher(None, self.hashed_features[hash], compared_features[hash]).ratio())
    #     # avg = 0 
    #     # sum = 0 
    #     # for i in sim_index:
    #     #     sum += i
    #     # avg = sum / len(sim_index) 
    #     sum = 1.5*sim_index[0] + 1.5*sim_index[1] +  1.5*sim_index[2] + sim_index[3]
    #     avg = sum /5.5
    #     return avg * 100

    def get_similarity_index(self, compared_features):
        sim_index = []
        for hash in compared_features:
            hamming_distance = hex_to_hash(compared_features[hash]) - hex_to_hash(self.hashed_features[hash])
            sim_index.append(1 - (hamming_distance / 256.0))
        sum = 1.5*sim_index[0] + 1.5*sim_index[1] +  1.5*sim_index[2] + sim_index[3]
        avg = sum /5.5
        return avg * 100



if __name__ == "__main__":
    if os.path.isfile('./Database/hash.json'):
        file = open("./Database/hash.json",)
        file_hash= json.load(file)
    else:
        file_hash = {}

    for filename in os.listdir("./Database/Songs1"):
        file={}
        song_path= os.path.join(curr_dir+"/Database/Songs", filename)
        song = Song(song_path)
        song.gen_spectrogram()
        song.save_spectrogram("./Database/spectrograms/")
        song.get_features()
        file.update({filename: song.features})
        song.write_json("./Database/features/"+filename+".json", file_hash)
        song.getHashedData(song.hashed_features, song.features)
        file_hash.update({filename: song.hashed_features})
        song.write_json("./Database/hash.json", file_hash)


