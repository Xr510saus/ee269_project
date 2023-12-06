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
# covid = []
sr = []

# counter = 0
# total = 0

# print('Accessing covid folder.')

# cfiles = glob.glob(path + 'covid/*')

# for i in range(1000):
#     data, rate = librosa.load(cfiles[i], sr=None)
#     cough_segments, cough_mask = segment_cough(data, rate)
#     for seg in cough_segments:
#         counter += 1
#         total += len(seg)

#     if i % 100 == 0:
#         print('Finished %d cfiles' % i)

# print('Accessing healthy folder.')

# hfiles = glob.glob(path + 'healthy/*')

# for i in range(1000):
#     data, rate = librosa.load(hfiles[i], sr=None)
#     cough_segments, cough_mask = segment_cough(data, rate)
#     for seg in cough_segments:
#         counter += 1
#         total += len(seg)

#     if i % 100 == 0:
#         print('Finished %d cfiles' % i)

# print('Average sample length: ', total / counter)

cfiles = glob.glob(path + 'test/*')
test = []

for i in range(2):#file in cfiles:
    data, rate = librosa.load(cfiles[i], sr=22050)
    # sf.write(cfiles[i][:-4] + '_22k.wav', data, rate)

    cough_segments, cough_mask = segment_cough(data, rate, cough_padding=0.1)
    for seg in cough_segments:
        temp = np.resize(seg, (22050))
        temp, _ = preprocess_cough(temp, rate)
        test.append(temp)

np.savetxt(path + 'test/test.txt', test)

temp = np.loadtxt(path + 'test/test.txt')
print(temp.shape)

    # ipd.Audio(data, rate=rate)
    # sr.append(rate)

# hfiles = glob.glob(path + 'healthy/*')
# for file in hfiles:
#     data, rate = librosa.load(file, sr=None)
#     sr.append(rate)

# print('Min sr: ', min(sr))