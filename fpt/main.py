from fpt import FptConfig, drs
from visualization import discrete
import serialization as ser
from matplotlib import pyplot as plt
from time import perf_counter
import numpy as np
from utilities_store import plot_spectrogram_arrows
from scipy.io import wavfile

plt.rcParams["font.family"] = "Helvetica"

if __name__ == '__main__':
    n0 = 0
    N = 512
    dir = "./data/input/london/violin"
    # filename = "violin_As5_long_forte_molto-vibrato"
    dir = "./data/input"
    # filename = "bach"
    filename = "flute-a4"
    source, sample_rate = ser.from_file(filename, dir=dir)
    
    signal = source # [len(source)//4:len(source)//4+N]
    step_size = N
    config = FptConfig(
        length=N,
        degree=N // 2, # K
        delay=0,
        sample_rate=sample_rate,
        power_threshold=1e-8,
        decay_threshold=0
    )
    run = False
    if run:
        start = perf_counter()
        spectrogram = drs(signal, config, step_size)
        print(perf_counter() - start)
        ser.save(spectrogram)
    spectrogram = ser.load() 
    
    ser.to_csv(spectrogram)
    # Commented because of pickle error
    plt.ion()
    recon = spectrogram.reconstruction
    # ser.to_wav(recon, filename, sample_rate)
    
    discrete(spectrogram, max_freq=22050)
    plt.show(block=True)
    # plot_spectrogram_arrows(spectrogram)
    #
    # fig, ax = plt.subplots()
    # ts = np.arange(len(signal)) / sample_rate
    # ax.plot(ts, signal, label="Original")
    # ts_ = np.arange(len(recon)) / sample_rate
    # plt.plot(ts_, spectrogram.reconstruction, label='Reconstruction', alpha=0.5)
    # plt.legend()
    # ax.set_xlim(ts[0], ts[-1])
    # ax.set_xlabel("Time (s)")
    # y_max = np.max(abs(signal)) * 1.15
    # ax.set_ylim(-y_max, y_max)
    # ax.set_ylabel("Amplitude")
    # fig.set_figwidth(10)
    # fig.set_figheight(10)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.grid(axis="y")
    # plt.savefig(f'plots/signal_{filename}.svg', format='svg', bbox_inches="tight")
    # plt.show()
    #
    # fig, ax = plt.subplots()
    # fig.set_figwidth(10)
    # fig.set_figheight(10)
    # ax.set_xlabel("Time (s)")
    # ax.set_ylabel("Frequency (Hz)")
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # spectrum, freqs, t, im = plt.specgram(x=signal, Fs=sample_rate, NFFT=N, noverlap=N - step_size,
    #                                       window=plt.mlab.window_none)
    # idx = min(np.count_nonzero(freqs <= 5000) + 1, len(freqs))
    # spectrum = 10 * np.log10(spectrum[:idx])
    # # spectrum = spectrum[:idx]
    # freqs = freqs[:idx]
    # t = np.insert(t, 0, 0)
    # im = ax.imshow(np.flipud(spectrum), aspect='auto',
    #                extent=(np.amin(t) * sample_rate, np.amax(t) * sample_rate, freqs[0], freqs[-1]))
    # # fig.colorbar(im, ax=ax, shrink=0.35, label="Power (dB)")
    # plt.yticks([0, 1000, 2000, 3000, 4000, 5000], ['0', '1K', '2K', '3K', '4K', '5K'])
    # txs = np.arange(0, t[-1], 0.1)
    # plt.xticks(txs * sample_rate, [f'{tx:0.1f}' for tx in txs])
    # ax.grid(axis="y")
    # plt.savefig(f'plots/fourier_{filename}.svg', format='svg', bbox_inches="tight")
    # plt.show()
    #
    # fig, ax = plt.subplots()
    # ax.set_xlabel('Time (s)')
    # ax.set_ylabel('Frequency (Hz)')
    # ax.set_ylim((0, 5000))
    # plt.yticks([0, 1000, 2000, 3000, 4000, 5000], ['0', '1K', '2K', '3K', '4K', '5K'])
    # # ax.set_title(title)
    # spectrogram = spectrogram[np.isfinite(spectrogram.power)]
    # power = spectrogram.power
    # normed = power / np.nanmax(power)
    # # unit = fig.bbox.width * fig.bbox.height / 10000
    #
    # ax.scatter(spectrogram.onsets / spectrogram.sample_rate, spectrogram.frequency, c=power,
    #            s=normed * 100)
    #
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.grid(axis="y")
    # fig.set_figwidth(10)
    # fig.set_figheight(10)
    # plt.savefig(f'plots/discrete_{filename}.svg', format='svg', bbox_inches="tight")
    # plt.show()
