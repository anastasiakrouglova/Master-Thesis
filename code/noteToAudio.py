import os, sys
# from os.path import exists
# from matplotlib import pyplot as plt
# import pandas as pd
from pathlib import Path  

## 0. Import the library of functions
sys.path.insert(0, './code/fpt')
from utilities_store import *
import serialization as ser
import pandas as pd
import soundfile as sf


filename = 'tester'
#path_input = 'fpt/data/input/scores/' #scores/
path_input = './code/fpt/data/input/' #scores/
path_output = './code/fpt/data/output/scores/'

file = path_output+filename+".csv"






# assume we have columns 'time' and 'value'
signal = pd.read_csv('./code/fpt/data/output/scores/flute_syrinx_1.csv')
sample_rate = 44100


sf.write(f'./lalala.wav', signal.real, sample_rate)


# # compute sample rate, assuming times are in seconds
# times = df['onset'].values
# n_measurements = len(times)
# timespan_seconds = times[-1] - times[0]
# sample_rate_hz = int(n_measurements / timespan_seconds)

# # write data
# data = df['frequency'].values
# sf.write('recording.wav', data, sample_rate_hz)





# def to_wav(signal, filename='flute-a4', sample_rate=44100, dir='./code/fpt/data/output/'):
#     sf.write(f'./lalala.wav', signal.real, sample_rate)
    

# to_wav