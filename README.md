# ShazamLikeApp
## **Description**
A basic implementation of a song  identification algorithm, it utilizes the advantages of a spectrogram and perceptual hashing, implementation is **as follows :**
1. A database is formed of multiple songs (Audio File) separated to their Vocal and Musical features.
2. Extraction of Spectrogram and spectral Features (Mel Spectrogram, Mel frequency Coefficient and Chroma STFT) is executed.
3. Hashing the extracted data with a Perceptual Hashing Algorithm.
4. A test Song (Audio File) is given to the application with extraction of its Hash the matches are found.
5. Matching percentages are calculated according to a mapping algorithm and then sorted to the user.

A testing mechanism is implemented by mixing two Audio files then this mix is given to the application to find it's matches in the database