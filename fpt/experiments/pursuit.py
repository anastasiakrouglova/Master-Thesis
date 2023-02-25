from fpt import from_file, FptConfig, drs, FptMinus, FptPlus
from resonance import save, load, ResonanceSpectrum, Resonance, ResonanceSpectrogram
from matplotlib import pyplot as plt
import numpy as np
from visualization import power_spectrum, discrete
from dataclasses import replace
import time


def power(signal: np.ndarray):
    return np.sum(np.abs(signal) ** 2) / len(signal)


def pursuit(signal: np.ndarray, config: FptConfig):
    threshold = 0  # 1e-4
    resonances = []
    combined = ResonanceSpectrum(np.array([]), config.length, config.sample_rate)
    K = config.degree
    flip = False
    fail = False
    while power(signal) > threshold and K < len(signal) // 2 and len(combined) < len(signal) // 2:
        # print(f"{'-->' if not flip else '<--'}\t{K}\t{power(signal):.6f}\t{len(combined)}")
        spectrum = FptPlus(signal, replace(config, degree=K)).spectrum.filter_decay()
        if len(spectrum) == 0 and fail:
            K *= 2
            fail = False
            flip = not flip
            signal = np.flip(signal)
        elif len(spectrum) == 0:
            fail = True
            flip = not flip
            signal = np.flip(signal)
        else:
            fail = False
            # max_resonance = spectrum.spectrum[np.argmax(spectrum.powers)]
            max_spectrum = spectrum[np.argsort(spectrum.amplitudes)[-K:]]
            spec = max_spectrum.mirror if flip else max_spectrum
            combined = combined + spec if len(combined) != 0 else spec
            signal = signal - spectrum.reconstruction
    return combined


if __name__ == '__main__':
    # signal, sample_rate = from_file("nine")
    # N = 1024
    # K = 2
    # S = 5
    # config = FptConfig(
    #     length=N,
    #     degree=N // 2,
    #     delay=0,
    #     sample_rate=sample_rate,
    #     # convergence_threshold=1e-3,
    #     # amplitude_threshold=1e-3,
    #     # power_threshold=1e-8,
    #     decay_threshold=0
    # )

    # forward = FptPlus(signal, config).spectrum.filter_decay()
    # backward = FptPlus(np.flip(signal), config).spectrum.filter_decay()
    # combined_spectrum = forward + backward.mirror
    #
    # plt.figure()
    # plt.plot(signal, label="Original")
    # plt.plot(combined_spectrum.reconstruction, label="Reconstruction")
    # plt.legend()
    # plt.show()
    #
    # plt.figure()
    # plt.plot((combined_spectrum.reconstruction - signal)**2)
    # plt.show()

    # config = FptConfig(
    #     length=N,
    #     degree=K,
    #     delay=0,
    #     sample_rate=sample_rate,
    # )

    # pursuit_spectrum = pursuit(signal[N * S:N * (S + 1)], config)
    # plt.figure()
    # plt.plot(signal[N * S:N * (S + 1)], label="Original")
    # plt.plot(pursuit_spectrum.reconstruction, label="Pursuit")
    # plt.show()

    # spectra = []
    # start = time.perf_counter()
    # for S in range(len(signal) // N):
    #     pursuit_spectrum = pursuit(signal[N * S:N * (S + 1)], config)
    #     spectra.append(pursuit_spectrum)
    # duration = time.perf_counter() - start
    # print(f"Duration: {duration}")
    # spectrogram = ResonanceSpectrogram(spectra, len(signal), sample_rate)
    # plt.figure()
    # plt.plot(signal, label="Original")
    # plt.plot(spectrogram.reconstruction, label="Reconstruction")
    # plt.legend()
    # plt.plot()

    N0 = 5000
    N = 2048
    inc = 1
    signal, sample_rate = from_file("nine")
    spectra = []
    for K in [n for n in range(2, N // 2 + 1, inc)]:
        config = FptConfig(
            length=N,
            degree=K,
            delay=0,
            sample_rate=sample_rate,
            # power_threshold=1e-6
        )
        spectrum = drs(signal[N0:N0 + N], config, N)[0]
        spectra.append(replace(spectrum, N=inc))
    spectrum_sequence = ResonanceSpectrogram(spectra, N // 2, 1)
    discrete(spectrum_sequence)
    plt.show()

    plt.figure()
    msrs = [np.sum(np.abs(spectrum.reconstruction - signal[N0:N0+N])**2)/N
            for spectrum in spectra]
    plt.plot(msrs)
    plt.show()

    # plt.plot((np.abs(signal - heavy_spectrum.reconstruction)) ** 2, label="heavy")
    # plt.plot(np.abs(signal - pursuit_spectrum.reconstruction) ** 2, label="pursuit")

    # for K in [2 ** n for n in range(1, 8)]:
    #     config = replace(config, degree=K)
    #     spectrum = FptMinus(signal, config).spectrum
    #     max_res = np.argmax(spectrum.amplitudes)
    #     plt.scatter(spectrum.frequencies[max_res], spectrum.amplitudes[max_res], label=K)
    # plt.legend()
    # plt.show()
