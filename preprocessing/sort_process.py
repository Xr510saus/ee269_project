import numpy as np
import pandas as pd
import os
import glob

import pickle
from scipy.io import wavfile
import librosa
from DSP import preprocess_cough
from segmentation import segment_cough
import IPython.display as ipd
import soundfile as sf

path = 'EE 269/dataset/coughvid_20211012/'

# Delete .jsons, .oggs, and .webms (post conversion to .wavs)
# for json in glob.glob(path + '*.json'):
#     os.remove(json)
# for ogg in glob.glob(path + '*.ogg'):
#     os.remove(ogg)
# for webm in glob.glob(path + '*.webm'):
#     os.remove(webm)

# Sort remaining (.wav) audio files into healthy or covid (includes symptomatic)
# df = pd.read_csv(path + 'metadata_compiled.csv')
# names = df.uuid.to_numpy()
# status = df.status.to_numpy()

# for i in range(len(names)):
#     file = names[i] + '.wav'

#     if status[i] == 'healthy':
#         if os.path.exists(path + file):
#             os.rename(path + file, path + 'healthy/' + file)
#     elif status[i] == 'symptomatic' or status[i] == 'COVID-19':
#         if os.path.exists(path + file):
#             os.rename(path + file, path + 'covid/' + file)
#     else:
#         if os.path.exists(path + file):
#             os.remove(path + file)

# Preprocess data to make same lengths and normalize

# Covid files
cfiles = glob.glob(path + 'covid/*')
cont = []

# To reduce amount of files needed to work with, grab the first 1000
for i in range(1000):
    data, rate = librosa.load(cfiles[i], sr=22050) # Resample each at 22050 Hz

    cough_segments, cough_mask = segment_cough(data, rate, cough_padding=0.1)

    for seg in cough_segments:
        crop = np.resize(seg, (22050)) # Resize so that each will be the same length
        proc, _ = preprocess_cough(crop, rate)
        cont.append(proc)

np.savetxt(path + 'covid.txt', cont)

# Healthy files

hfiles = glob.glob(path + 'healthy/*')
cont = []

# To reduce amount of files needed to work with, grab the first 1000
for i in range(1000):
    data, rate = librosa.load(hfiles[i], sr=22050) # Resample each at 22050 Hz

    cough_segments, cough_mask = segment_cough(data, rate, cough_padding=0.1)

    for seg in cough_segments:
        crop = np.resize(seg, (22050)) # Resize so that each will be the same length
        proc, _ = preprocess_cough(crop, rate)
        cont.append(proc)

np.savetxt(path + 'healthy.txt', cont)