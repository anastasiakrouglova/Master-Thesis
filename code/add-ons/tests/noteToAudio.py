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
import csv


filename = 'tester'
#path_input = 'fpt/data/input/scores/' #scores/
path_input = './code/fpt/data/input/' #scores/
path_output = './code/fpt/data/output/scores/'

file = path_output+filename+".csv"

# assume we have columns 'time' and 'value'
# signal = pd.read_csv('./code/fpt/data/output/scores/flute_syrinx_1.csv')
# signal = pd.read_csv('./code/fpt/data/output/flute-a4_f0.csv')
ACCURACY = 650  # Generally a good value for most musical analysis
MAX_FREQ = 2000


noisy_spectrogram, noisy_signal = get_spectrogram(path_input+filename+'.wav', N = ACCURACY, step_size = 250, 
                                            power_threshold = 1e-9, amp_threshold = 1e-8)


print(noisy_signal)


sample_rate = 44100


# with open(file, 'r') as file:
#   csvreader = csv.reader(file)
# #   print(signal)
#   ser.from_csv(csvreader)

# spectrogram = ser.from_csv(signal)
# print(spectrogram)


# ser.from_csv(file)

# with open(file, newline='') as file:
#     reader = csv.reader(file)
#     # print(type(list(reader)[1][2]))
#     # print(len(list(reader)))
#     for i in range(len(list(reader))):
#         print([complex(x) for x in list(reader)[i]])
#         print('a')
    
    
    # type(list(reader)[1][2])
    # print([float(x) for x in a])
    # print(a)
    # a = np.array(list(reader)[1], dtype=float)
    #print(a)
    #return np.array(list(reader), dtype=float)


# sf.write(f'./lalala.wav', signal.real, sample_rate)
# ipd.Audio(data = noisy_signal, rate = noisy_spectrogram.sample_rate[0])
# ipd.Audio(data = denoised_spectrogram.reconstruction.real, rate = noisy_spectrogram.sample_rate[0])

#write("nine-denoised.wav", noisy_spectrogram.sample_rate[0], denoised_spectrogram.reconstruction.real)

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