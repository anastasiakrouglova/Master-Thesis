from resonance import ResonanceSet
import numpy as np
import matplotlib.pyplot as plt


def threedee(spectrogram: ResonanceSet):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    onsets, frequencies, powers = np.array(list(zip(*[
        (onset, frequency, power)
        for onset, spectrum in zip(spectrogram.onsets, spectrogram.spectra)
        for frequency, power in zip(spectrum.frequencies, spectrum.powers)
        if frequency >= 0
    ])))
    ax.scatter(onsets / spectrogram.sample_rate, frequencies, powers, c=powers)
    ax.set_ylim(0, 7000)


def discrete(spectrogram, signal=None, max_freq=3000, params=None):
    # if params is not None:
    #     title = ' '.join([(f'{k}={v}') for k, v in params.items()])
    # else:
    #     title = 'Resonance Spectrogram'
    fig, ax = plt.subplots()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_ylim((0, max_freq))
    # ax.set_title(title)
    spectrogram = spectrogram[np.isfinite(spectrogram.power)]
    normed = spectrogram.power / np.nanmax(spectrogram.power)
    # unit = fig.bbox.width * fig.bbox.height / 10000
    unit = 1280 * 960 / 20000

    ax.scatter(spectrogram.onsets / spectrogram.sample_rate, spectrogram.frequency, c=spectrogram.power,
               s=unit * normed)

    def on_resize(event):
        ax.clear()
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_ylim((0, max_freq))
        # ax.set_title(title)
        unit = event.width * event.height / 20000
        ax.scatter(spectrogram.onsets / spectrogram.sample_rate, spectrogram.frequency, c=spectrogram.power,
                   s=unit * normed)

    ax.figure.canvas.mpl_connect('resize_event', on_resize)


def discrete_old(spectrogram: ResonanceSet, mode: str = "power"):
    value_fn = {
        "power": lambda spectrum: spectrum.power,
        "amplitude": lambda spectrum: spectrum.amplitude,
        "harmonic": lambda spectrum: spectrum.harmonic_weight,
        "height": lambda spectrum: spectrum.height,
        "area": lambda spectrum: spectrum.area,
    }[mode]

    onsets, frequencies, values = np.array(list(zip(*[
        (onset, frequency, value)
        for onset, spectrum in zip(spectrogram.onsets, spectrogram.spectra)
        for frequency, value in zip(spectrum.frequency, value_fn(spectrum))
        if frequency >= 0
    ])))
    fig, ax = plt.subplots()

    duration = spectrogram.max_duration / spectrogram.sample_rate
    onsets = onsets / spectrogram.sample_rate
    mask = np.isfinite(values)
    onsets, frequencies, values = onsets[mask], frequencies[mask], values[mask]
    normed_values = values / np.max(values)
    # unit = fig.bbox.width * fig.bbox.height / 10000
    unit = 1280 * 960 / 10000
    ax.scatter(onsets, frequencies, c=values, s=unit * normed_values)
    ax.set_xlim(0, duration)
    ax.set_ylim(0, 5000)
    ax.set_title(f"Discrete Resonance Spectrogram (Mode: {mode})")

    def on_resize(event):
        ax.clear()
        unit = event.width * event.height / 10000
        ax.scatter(onsets, frequencies, c=values, s=unit * normed_values)
        ax.set_xlim(0, duration)
        ax.set_ylim(0, 5000)

    ax.figure.canvas.mpl_connect('resize_event', on_resize)


def continuous(spectrogram: ResonanceSet):
    onsets = spectrogram.onsets / spectrogram.sample_rate
    frequencies = np.linspace(0, 10000, 100)
    bins = [[spectrum.bin(f0, f1) for spectrum in spectrogram]
            for f0, f1 in zip(frequencies, frequencies[1:])]
    plt.contourf(onsets, frequencies[:-1], np.log(bins),
                 cmap='viridis', levels=100)
    plt.xlim(0, np.max(onsets))
    plt.ylim(0, np.max(frequencies))
    plt.title('Continuous Resonance Spectrogram')


def power_spectrum(spectrum: ResonanceSet, label=None):
    if len(spectrum) == 0:
        return
    frequencies = np.linspace(0, 5000, 10000)
    plt.plot(frequencies,
             np.log(np.abs(np.sum(spectrum.map(lambda _, res: res.at(frequencies)), axis=0) ** 2)),
             label=label)
    plt.title('Power Spectrum')
