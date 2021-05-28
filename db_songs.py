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
import difflib




logger = logging.getLogger()

curr_dir = os.getcwd()

class Song():
    def __init__(self,path= ""):
        self.path=path
        self.data = None
        self.sr = None
        self.hop_length = 512 #50% overlap
        self.window_size = 1024
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
        self.features["mel_spectrogram"] = librosa.feature.melspectrogram(self.data, sr=self.sr, window='hann')
        #self.features["mel_spectrogram"] = librosa.power_to_db(S, ref=np.max)

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

    # def get_peaks(self):
    #     struct = generate_binary_structure(2, 1)
    #     neighborhood = iterate_structure(struct,10)
    #     local_max = maximum_filter(self.mel_spectrogram, footprint=neighborhood) == self.mel_spectrogram
    #     background = (self.mel_spectrogram == 0)
    #     eroded_background = binary_erosion(background, structure=neighborhood)
        
    #     #applying XOR between the matrices to get the boolean mask of spectrogram
    #     detected_peaks = local_max ^ eroded_background
        
    #     # extract peaks
    #     amps = self.mel_spectrogram[detected_peaks].flatten()
    #     peak_freqs , peak_times = np.where(detected_peaks)
        
    #     filtered_peaks = np.where(abs(amps) > 10)  # freq, time, amp
        
    #     #get freqs , times of the indicies
    #     freqs=peak_freqs[filtered_peaks]
    #     times=peak_times[filtered_peaks]
        
    #     self.peaks = list(zip(freqs, times))

    #     #print(len(list(zip(self.peaks))))

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

    # gets the similarity index between this song and another song features
    def get_similarity_index(self, compared_features):
        sim_index = []
        for hash in compared_features:
            sim_index.append(difflib.SequenceMatcher(None, self.hashed_features[hash], compared_features[hash]).ratio())
        # avg = 0 
        # sum = 0 
        # for i in sim_index:
        #     sum += i
        # avg = sum / len(sim_index) 
        sum = sim_index[1] + sim_index[2]
        avg = sum / 2
        return avg * 100

if __name__ == "__main__":
    file_hash = {}
    for filename in os.listdir("./Database/Songs"):
        #file={}
        song_path= os.path.join(curr_dir+"/Database/Songs", filename)
        song = Song(song_path)
        song.gen_spectrogram()
        #song.save_spectrogram("./Database/spectrograms/")
        song.get_features()
        #file.update({filename: song.features})
        #write_json("./Database/feautures"+filename+".json", file_hash)
        song.getHashedData(song.hashed_features, song.features)
        file_hash.update({filename: song.hashed_features})
        song.write_json("./Database/hash.json", file_hash)


