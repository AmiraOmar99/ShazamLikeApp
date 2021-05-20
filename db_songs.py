import os
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import librosa.display
from librosa.core import load
from pydub import AudioSegment
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import (generate_binary_structure, iterate_structure, binary_erosion)

curr_dir = os.getcwd()


class Song():
    def __init__(self,path= ""):
        self.path=path
        self.data = None
        self.sr = None
        self.hop_length = 2048 #50% overlap
        self.window_size = 4096
        self.mel_spectrogram =None
        self.mfcc = None
        self.chroma_stft = None
        self.onset_frames= None
        self.peaks=None
        self.read_song()
        
        
   

    def read_song(self):
        self.name, ext = os.path.splitext(self.path)
        
        if ext in [".mp3",".MP3"]:
            sound = AudioSegment.from_mp3(self.path)
            sound.export("converted.wav", format="wav")
            path = "./converted.wav"

        self.data, self.sr = librosa.core.load(path, mono=True, duration=60)
        pass

    def gen_spectrogram(self):
        S = librosa.feature.melspectrogram(self.data, sr=self.sr, n_fft=self.window_size, hop_length=self.hop_length, n_mels=128)
        self.mel_spectrogram = librosa.power_to_db(S, ref=np.max)


    def save_spectrogram(self,path):
        fig = plt.Figure()       
        ax = fig.add_subplot(111)
        librosa.display.specshow(self.mel_spectrogram, sr=self.sr, hop_length=self.hop_length, x_axis='time', y_axis='mel')
        plt.savefig(path+os.path.split(self.name)[1])
 
    def features(self):
        self.mfcc = librosa.feature.mfcc(y=self.data.astype('float64'), sr=self.sr)
        self.chroma_stft =librosa.feature.chroma_stft(y= self.data, sr=self.sr)
        #self.tempogram=librosa.feature.tempogram()
        o_env = librosa.onset.onset_strength(self.data, sr=self.sr)
        #times = librosa.times_like(o_env, sr=self.sr)
        self.onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=self.sr)
        #self.peaks=librosa.util.peak_pick(x, pre_max, post_max, pre_avg, post_avg, delta, wait)
        #self.get_peaks()

        
    def get_peaks(self):
        struct = generate_binary_structure(2, 1)
        neighborhood = iterate_structure(struct,10)
        local_max = maximum_filter(self.mel_spectrogram, footprint=neighborhood) == self.mel_spectrogram
        background = (self.mel_spectrogram== 0)
        eroded_background = binary_erosion(background, structure=neighborhood)
        
        #applying XOR between the matrices to get the boolean mask of spectrogram
        detected_peaks = local_max ^ eroded_background
        
        # extract peaks
        amps = self.mel_spectrogram[detected_peaks].flatten()
        peak_freqs , peak_times = np.where(detected_peaks)
        
        filtered_peaks = np.where(abs(amps) > 10)  # freq, time, amp
        
        #get freqs , times of the indicies
        freqs=peak_freqs[filtered_peaks]
        times=peak_times[filtered_peaks]
        
        self.peaks = list(zip(freqs, times))

        #print(len(list(zip(self.peaks))))








if __name__ == "__main__":
    for filename in os.listdir(".\Database\Songs"):
        song_path= os.path.join(curr_dir+"\Database\Songs", filename)
        #print(song_path)
        song = Song(song_path)
        song.gen_spectrogram()
        song.save_spectrogram(".\Database\spectrograms/")
        song.features()
