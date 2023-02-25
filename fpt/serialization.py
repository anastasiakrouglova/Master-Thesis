import pickle
import soundfile as sf
import resonance
import csv
import numpy as np


def load(filename='temp', dir='./data/pickles'):
    obj = pickle.load(open(f'{dir}/{filename}.pkl', 'rb'))
    return obj


def save(obj, filename='temp', dir='./data/pickles'):
    pickle.dump(obj, open(f'{dir}/{filename}.pkl', 'wb'))


def from_file(filename, dir='./data/input', filetype='wav'):
    file_in = f'{dir}/{filename}.{filetype}'
    signal, sample_rate = sf.read(file_in)
    if len(signal.shape) > 1:
        signal = signal[:, 0]  # turn stereo into mono
    return signal, sample_rate


def to_wav(signal, filename='temp', sample_rate=44100, dir='./data/output'):
    sf.write(f'{dir}/{filename}.wav', signal.real, sample_rate)


def to_csv(s, filename='temp', dir='./data/output'):
    print('Writing csv...', end='')
    with open(f'{dir}/{filename}.csv', 'w', newline='') as file:
        row = ('onset', 'duration', 'sample_rate',
               'amplitude', 'phase', 'frequency', 'decay',
               'd', 'w', 'z')
        rows = list(zip(s.onsets, s.N, s.sample_rate,
                        s.amplitude, s.phase, s.frequency, s.decay,
                        s.d, s.w, s.z))
        writer = csv.writer(file)
        writer.writerow(row)
        writer.writerows(rows)
    print('Done.')


def from_csv(filename):
    with open(filename, newline='') as file:
        reader = csv.reader(file)
        return np.array(list(reader), dtype=float)
