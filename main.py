import Shazam_ui
import PyQt5.QtGui
from PyQt5 import  QtWidgets
from PyQt5.QtWidgets import QFileDialog
from db_songs import Song
import librosa.display
import numpy as np
import sys
import os
import scipy.io.wavfile as wavf
import logging
import json

#Create and configure logger
logging.basicConfig(filename="logging.log", format='%(asctime)s %(message)s',filemode='w')

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

class MainWindow(QtWidgets.QMainWindow, Shazam_ui.Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.song1 = None
        self.song2 = None
        self.songMix = None
        self.songs=[self.song1,self.song2]
        self.song_labels=[self.song1_name,self.song2_name]
        self.slider.setDisabled(True)

        #connections
        self.open_song1.clicked.connect(lambda: self.open(0))
        self.open_song2.clicked.connect(lambda: self.open(1))
        self.slider.valueChanged.connect(lambda: self.mix())
        self.Recognize.clicked.connect(lambda: self.check_opened(0))
        self.Recognize_2.clicked.connect(lambda: self.check_opened(1))

    def open(self,index):
        logger.debug('open song {}'.format(index+1))
        song_path = PyQt5.QtWidgets.QFileDialog.getOpenFileName(None, 'open song', None, "MP3 *.mp3;; WAV *.wav")[0]
        if song_path:
            self.songs[index]=Song(song_path)
            self.songs[index].gen_spectrogram()
            self.songs[index].get_features()
            self.songs[index].getHashedData(self.songs[index].hashed_features,self.songs[index].features)
            # self.songs[index].hashed_features["mel_spectrogram"]=self.createPerceptualHash(np.array(self.songs[index].features["mel_spectrogram"]))
            # self.songs[index].hashed_features["mfcc"]=self.createPerceptualHash(np.array(self.songs[index].features["mfcc"]))
            # self.songs[index].hashed_features["chroma_stft"]=self.createPerceptualHash(np.array(self.songs[index].features["chroma_stft"]))
            # self.songs[index].hashed_features["onset_frames"]=self.createPerceptualHash(np.array(self.songs[index].features["onset_frames"]))

            #print(self.songs[index].features["mel_spectrogram"][0][0:30])
            #print(self.songs[index].hashed_features["mel_spectrogram"])
            #print(self.songs[index].hashed_features["mfcc"])

        for song in self.songs:
            name=song_path.split("/")[-1]
            self.song_labels[index].setText(name)
        #print(len(self.songs[index].data))
        self.mix()

    def mixSongs(self,song1, song2, ratio):
        logger.debug("Mixing the 2 songs together with {} % of song1".format(ratio*100) + " and {} % of song2".format((1-ratio)*100) )  
        return (ratio*song1 + (1-ratio)*song2)

    def mix(self):
            logger.debug("Checking if the 2 songs are opened")  
            if self.songs[0] != None and self.songs[1] != None:
                self.slider.setEnabled(True)
                self.slider_percentage.setText(str(self.slider.value())+"%")
                self.songMix=self.mixSongs(self.songs[0].data, self.songs[1].data,self.slider.value()/100)
                wavf.write("mix.wav", 22050 , self.songMix)
                self.mixFile=Song("mix.wav")
                self.mixFile.gen_spectrogram()
                self.mixFile.get_features()
                self.mixFile.getHashedData(self.mixFile.hashed_features,self.mixFile.features )
                # print(self.mixFile.hashed_features)
                self.compare_songs(self.mixFile)
                # self.mixFile.hashed_features["mel_spectrogram"]=self.createPerceptualHash(np.array(self.mixFile.features["mel_spectrogram"]))
                # self.mixFile.hashed_features["mfcc"]=self.createPerceptualHash(np.array(self.mixFile.features["mfcc"]))
                # self.mixFile.hashed_features["chroma_stft"]=self.createPerceptualHash(np.array(self.mixFile.features["chroma_stft"]))
                # self.mixFile.hashed_features["onset_frames"]=self.createPerceptualHash(np.array(self.mixFile.features["onset_frames"]))

                #print(len(self.mixFile.features))
                #print(len(self.songMix.data))

                logger.debug("Mixing Output") 
            else:
                self.slider.setDisabled(True)

    def check_opened(self,index):
        if(self.songs[index]):
            self.compare_songs(self.songs[index])
        else:
            msg = PyQt5.QtWidgets.QMessageBox()
            msg.setWindowTitle('ERROR')
            msg.setText('Error: please select a song to recognize.')
            msg.setIcon(PyQt5.QtWidgets.QMessageBox.Critical)
            msg.exec_()

    # def compare_songs(self, mysong):
    #     self.similarity_indices = {}
    #     logger.debug("Comparing song with the database songs")
    #     for filename in os.listdir("./Database/features"):
    #         file = open("./Database/features/"+filename,)
    #         data = json.load(file)
    #         features = data[filename[:-4]+"mp3"]
    #         self.similarity_indices[filename[:-5]] = mysong.get_similarity_index(features)
    #         file.close()
    #     self.similarity_indices = dict(sorted(self.similarity_indices.items(), key=lambda item: item[1] , reverse=True))
    #     # print(self.similarity_indices)
    #     self.show_results() 

    def compare_songs(self, mysong):
        self.similarity_indices = {}
        logger.debug("Comparing song with the database songs")
        file = open("./Database/hash.json",)
        data = json.load(file)
        for song in data:
            self.similarity_indices[song[:-4]] = mysong.get_similarity_index(data[song])
        file.close()
        self.similarity_indices = dict(sorted(self.similarity_indices.items(), key=lambda item: item[1] , reverse=True)[:10])
        # print(self.similarity_indices)
        self.show_results() 


    def show_results(self):
        self.resultsTable.setColumnCount(2)
        self.resultsTable.setRowCount(len(self.similarity_indices))
        for index, key in enumerate(self.similarity_indices):
            self.resultsTable.setItem(index, 0, QtWidgets.QTableWidgetItem(key))
            self.resultsTable.setItem(index, 1, QtWidgets.QTableWidgetItem(str(round(self.similarity_indices[key], 2))+"%"))
            self.resultsTable.verticalHeader().setSectionResizeMode(index, QtWidgets.QHeaderView.Stretch)

        self.resultsTable.setHorizontalHeaderLabels(["Song Matches", "Similarity %"])

        for col in range(2):
            self.resultsTable.horizontalHeader().setSectionResizeMode(col, QtWidgets.QHeaderView.Stretch)

        self.resultsTable.show()    

    # def createPerceptualHash(self, feature):
    #     logger.debug("Creating Perceptual Hash for each feature")
    #     dataInstance = Image.fromarray(feature)
    #     hashed_data = imagehash.phash(dataInstance, hash_size=16)
    #     print(hashed_data)
    #     return hashed_data

    # def getHashedData(self, Hdic , Fdic):
    #     for key in Hdic:
    #         Hdic[key].append=self.createPerceptualHash(np.array(Fdic[key]))

         
if __name__ == "__main__":
    
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())