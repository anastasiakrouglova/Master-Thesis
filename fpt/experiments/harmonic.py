import resonance
import numpy as np
import serialization as ser

from fpt import FptConfig, drs
import visualization as viz
from matplotlib import pyplot as plt

from time import perf_counter
from dataclasses import replace
import os


def fundamental(res: resonance.Resonance, spectrum: resonance.ResonanceSet, h_max=25):
    harmonics = resonance.ResonanceSet(res.harmonics(h_max), np.array([0] * h_max))
    harmonicity = (spectrum @ harmonics).real / (spectrum.norm * harmonics.norm)
    return harmonicity


# Decaying amplitude
def f0(w, spectrum: resonance.ResonanceSet):
    da, wa, fa = spectrum.d.reshape(-1, 1), spectrum.w.reshape(-1, 1), spectrum.sample_rate.reshape(-1, 1)
    db, wb, fb = spectrum.d.reshape(1, -1), spectrum.w.reshape(1, -1), spectrum.sample_rate.reshape(1, -1)
    ds = da * db.conjugate()
    ws = wa - wb.conjugate()
    ts = -1 / (wa * np.tan(np.pi * wa / w)) + 1 / (wb.conjugate() * np.tan(np.pi * wb.conjugate() / w))
    return np.sign(w) * np.sum((ds / ws) * ts)


def eip(w, spectrum: resonance.ResonanceSet):
    dj, wj, fa = spectrum.d.reshape(-1, 1), \
                 spectrum.w.reshape(-1, 1), \
                 spectrum.sample_rate.reshape(-1, 1)
    dk, wk, fb = spectrum.d.reshape(1, -1).conjugate(), \
                 spectrum.w.reshape(1, -1).conjugate(), \
                 spectrum.sample_rate.reshape(1, -1)
    ds = dj * dk
    ws = wj - wk
    tj = np.pi / (w * np.tan(np.pi * wj / w)) + np.pi / (wj * np.tan(np.pi * w / wj))
    tk = np.pi / (w * np.tan(np.pi * wk / w)) + np.pi / (wk * np.tan(np.pi * w / wk))
    p = ds / ((w - wj) * (w - wk))
    total = np.sum((ds / ws) * (-tj + tk) + p)
    return total


# Constant Amplitude
# def f1(w, spectrum: resonance.ResonanceSet):
#     da, wa, fa = spectrum.d.reshape(-1, 1), spectrum.w.reshape(-1, 1), spectrum.sample_rate.reshape(-1, 1)
#     db, wb, fb = spectrum.d.reshape(1, -1), spectrum.w.reshape(1, -1), spectrum.sample_rate.reshape(1, -1)
#     ds = da * db.conjugate()
#     ws = wa - wb.conjugate()
#     ts = -1 / (w * np.tan(np.pi * wa / w)) + 1 / (wb * np.tan(np.pi * wb.conjugate() / w))
#     return np.sum((ds / ws) * ts)


if __name__ == '__main__':
    name = 'cello_E2_1_pianissimo_arco-normal'
    # name = 'cello_E2_1_forte_arco-normal'
    # name = 'cello_E4_1_forte_arco-normal'
    # name = 'cello_E4_1_pianissimo_arco-normal'
    # name = 'flute_D4_05_forte_normal'
    dir = "cello"
    source, sample_rate = ser.from_file(name, dir=f'../data/input/london/{dir}')

    h_max = 25
    n0 = len(source) // 2
    N = 4096
    step_size = N
    config = FptConfig(
        length=N,
        degree=N // 2,
        delay=0,
        sample_rate=sample_rate,
        power_threshold=0,
        decay_threshold=0
    )

    freqs = np.linspace(100, 1200 * 2 * np.pi, 100000)

    signal = source[n0:n0 + N]

    # signal_0 = np.cos(2 * np.pi * (330 + 1j) * np.arange(N)/sample_rate)
    # signal_1 = 100*np.cos(2 * np.pi * (2*330+5 + 10j) * np.arange(N)/sample_rate)
    # signal = signal_0 + signal_1

    spectrum_ = drs(signal, config, step_size)
    spectrum = spectrum_.filter(lambda _, res: res.frequency > 0)
    spectrum = spectrum.transform(lambda _, res: res.shift(N // 2).truncate(1))
    salience = np.array([eip(res.w.real, spectrum)
                         for res in spectrum.elements])
    # salience = eip(freqs, spectrum)
    # salience = np.array([eip(freq, spectrum)
    #                      for freq in freqs])
    # salience = f0(freqs, spectrum)
    # salience[np.isclose(spectrum.frequency, 0)] = 0
    # salience[np.isclose(spectrum.frequency, sample_rate / 2)] = 0
    # salience[np.isclose(spectrum.frequency, -sample_rate / 2)] = 0
    # salience[np.isclose(salience, np.inf)] = 0
    # salience[np.isclose(salience, -np.inf)] = 0
    # max_res = spectrum.elements[np.argmax(salience)]
    # f0 = abs(max_res.frequency)

    ax = plt.subplot()
    ax2 = ax.twinx()
    ax.scatter(spectrum.frequency, salience, color="tab:blue", alpha=0.8)
    # ax.scatter(freqs / (2 * np.pi), salience, color="tab:blue", alpha=0.8)
    ax2.scatter(spectrum.frequency, spectrum.power, color="tab:orange", marker=".", alpha=0.8)
    plt.show()
