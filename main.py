import Shazam
import PyQt5.QtGui
from PyQt5 import  QtWidgets
from PyQt5.QtWidgets import QFileDialog
from db_songs import Song
import sys
import os
import logging

#Create and configure logger
logging.basicConfig(filename="logging.log", format='%(asctime)s %(message)s',filemode='w')

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)




class MainWindow(QtWidgets.QMainWindow, Shazam.Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.song1 = None
        self.song2 = None
        self.songs=[self.song1,self.song2]


        #connections
        self.open_song1.clicked.connect(lambda: self.open(0))
        self.open_song2.clicked.connect(lambda: self.open(1))

    def open(self,index):
        logger.debug('open song {}'.format(index+1))
        song_path = PyQt5.QtWidgets.QFileDialog.getOpenFileName(None, 'open song', None, "MP3 *.mp3;; WAV *.wav")[0]
        if song_path:
            self.songs[index]=Song(song_path)
            self.songs[index].gen_spectrogram()
            self.songs[index].get_features()
            print(len(self.songs[index].features)) 


if __name__ == "__main__":
    
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())