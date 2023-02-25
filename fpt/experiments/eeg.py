import scipy.io
import scipy.stats
import scipy.linalg
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, '/Users/nastysushi/Mirror/_MULTIMEDIA/THESIS/resonances/fpt-master')

from fpt import FptConfig, drs


import serialization as ser
import visualization as viz

# 0.2039475 to 0.4013158
# 0.5986842 to 0.7960526

if __name__ == '__main__':
    audio_1, sr1 = ser.from_file('Brainstem Stimulus 1', dir="../data/eeg_data")
    audio_2, sr2 = ser.from_file('Brainstem Stimulus 2', dir="../data/eeg_data")
    mastoids_1 = ser.load('mastoids_1', dir="../data/eeg_data")
    mastoids_2 = ser.load('mastoids_2', dir="../data/eeg_data")
    # allchannels_1 = ser.load('allchannels_1', dir="../data/eeg_data")
    # allchannels_2 = ser.load('allchannels_2', dir="../data/eeg_data")

    mastoids_1_exg1 = mastoids_1.transpose()[1]
    mastoids_1_exg2 = mastoids_1.transpose()[2]

    mastoids_2_exg1 = mastoids_2.transpose()[1]
    mastoids_2_exg2 = mastoids_2.transpose()[2]

    N = 1024
    step_size = N // 2
    sample_rate = 16384

    config = FptConfig(
        length=N,
        degree=N // 2,
        delay=0,
        sample_rate=sample_rate,
        power_threshold=0,
        decay_threshold=0
    )

    print("Start Mastoids 1 EXG1")
    spectrogram = drs(mastoids_1_exg1, config, step_size)
    ser.save(spectrogram, filename=f"mastoids_1_exg1_N={N}_S={step_size}", dir="../data/output")
    ser.to_csv(spectrogram, filename=f"mastoids_1_exg1_N={N}_S={step_size}", dir="../data/output")
    print("Finish Mastoids 1 EXG1")
    del spectrogram

    print("Start Mastoids 1 EXG2")
    spectrogram = drs(mastoids_1_exg2, config, step_size)
    ser.save(spectrogram, filename=f"mastoids_1_exg2_N={N}_S={step_size}", dir="../data/output")
    ser.to_csv(spectrogram, filename=f"mastoids_1_exg2_N={N}_S={step_size}", dir="../data/output")
    print("Finish Mastoids 1 EXG2")
    del spectrogram

    print("Start Mastoids 2 EXG1")
    spectrogram = drs(mastoids_2_exg1, config, step_size)
    ser.save(spectrogram, filename=f"mastoids_2_exg1_N={N}_S={step_size}", dir="../data/output")
    ser.to_csv(spectrogram, filename=f"mastoids_2_exg1_N={N}_S={step_size}", dir="../data/output")
    print("Finish Mastoids 2 EXG1")
    del spectrogram

    print("Start Mastoids 2 EXG2")
    spectrogram = drs(mastoids_2_exg2, config, step_size)
    ser.save(spectrogram, filename=f"mastoids_2_exg2_N={N}_S={step_size}", dir="../data/output")
    ser.to_csv(spectrogram, filename=f"mastoids_2_exg2_N={N}_S={step_size}", dir="../data/output")
    print("Finish Mastoids 2 EXG2")
    del spectrogram

    print("Start Mastoids 1 EXG1-EXG2")
    spectrogram = drs(mastoids_1_exg1 - mastoids_1_exg2, config, step_size)
    ser.save(spectrogram, filename=f"mastoids_1_exg1-exg2_N={N}_S={step_size}", dir="../data/output")
    ser.to_csv(spectrogram, filename=f"mastoids_1_exg1-exg2_N={N}_S={step_size}", dir="../data/output")
    print("Finish Mastoids 1 EXG1-EXG2")
    del spectrogram

    print("Start Mastoids 2 EXG1-EXG2")
    spectrogram = drs(mastoids_2_exg1 - mastoids_2_exg2, config, step_size)
    ser.save(spectrogram, filename=f"mastoids_2_exg1-exg2_N={N}_S={step_size}", dir="../data/output")
    ser.to_csv(spectrogram, filename=f"mastoids_2_exg1-exg2_N={N}_S={step_size}", dir="../data/output")
    print("Finish Mastoids 2 EXG1-EXG2")
    del spectrogram

    print("Start Mastoids 1 EXG1+EXG2")
    spectrogram = drs(mastoids_1_exg1 + mastoids_1_exg2, config, step_size)
    ser.save(spectrogram, filename=f"mastoids_1_exg1+exg2_N={N}_S={step_size}", dir="../data/output")
    ser.to_csv(spectrogram, filename=f"mastoids_1_exg1+exg2_N={N}_S={step_size}", dir="../data/output")
    print("Finish Mastoids 1 EXG1+EXG2")
    del spectrogram

    print("Start Mastoids 2 EXG1+EXG2")
    spectrogram = drs(mastoids_2_exg1 + mastoids_2_exg2, config, step_size)
    ser.save(spectrogram, filename=f"mastoids_2_exg1+exg2_N={N}_S={step_size}", dir="../data/output")
    ser.to_csv(spectrogram, filename=f"mastoids_2_exg1+exg2_N={N}_S={step_size}", dir="../data/output")
    print("Finish Mastoids 2 EXG1+EXG2")
    del spectrogram
