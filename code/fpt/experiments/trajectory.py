import numpy as np
import matplotlib.pyplot as plt

from fpt import FptConfig, bifpt
from resonance import Resonance, ResonanceSet


def plot_signal(signal, title="", plot_only=None):
    plt.figure()
    if plot_only is None or plot_only == "Real":
        plt.plot(signal.real, color="tab:blue", label="Real")
    if plot_only is None or plot_only == "Imag":
        plt.plot(signal.imag, color="tab:orange", label="Imag")
    if plot_only is None or plot_only == "Abs":
        plt.plot(abs(signal), color="tab:green", label="Abs")
        plt.plot(-abs(signal), color="tab:green")
    plt.xlim(0, N)
    plt.legend()
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    # Construct input resonance
    N = 1000
    sample_rate = 44100
    amplitude = 1000
    phase = np.pi / 3
    frequency = 1000
    decay = -100
    dk = amplitude * np.exp(1j * phase)
    wk = frequency + 1j * decay
    zk = np.exp(-1j * wk / sample_rate)
    input_res = Resonance(dk, wk, zk, 1000, sample_rate)
    input_res_conj = input_res.conjugate

    # Generate signal from input resonance
    input_signal = input_res.reconstruction #+ input_res_conj.reconstruction
    # plot_signal(input_signal, "Input", plot_only="Real")

    # Rotation in this manner only works for pure real signals
    #  Better approach is to use a pair for transformation matrices/vectors
    #  The matrix handles the rotations and dilations (multiplication)
    #  The vector handles the translations (addition)
    input_signal = input_res.reconstruction #+ input_res_conj.reconstruction
    # rotation = np.exp(1j * 0 * np.pi / 6)
    # rotated_time = np.arange(N) * rotation
    # rotated_signal = input_signal * rotation
    # rotated = (rotated_time + 1j * rotated_signal)  # / (np.arange(N) + 1)
    # plt.figure()
    # plt.plot(input_signal)
    # plt.plot(rotated.real)
    # plt.plot(rotated.imag)
    # plt.plot(rotated.real, rotated.imag)
    # plt.show()

    # Analyze input signal with FPT
    config = FptConfig(
        length=N,
        degree=N // 2,
        sample_rate=sample_rate,
        power_threshold=0,
    )
    # output_res = bifpt(input_signal, config)[0]

    # Generate signal from output resonance
    # output_signal = output_res.reconstruction
    # plot_signal(output_signal, "Output")

    # Create parametric curve from input signal
    xt = np.arange(N)
    yt = input_signal

    # Analyze parametric curve with FPT
    res_x = bifpt(xt, config)
    res_y = bifpt(yt, config)
    ids_x = np.array([hash(res) for res in res_x])
    ids_y = np.array([hash(res) for res in res_y])
    onsets_x = np.array([0] * len(ids_x))
    onsets_y = np.array([0] * len(ids_y))
    spec_x = ResonanceSet(ids_x, onsets_x)
    spec_y = ResonanceSet(ids_y, onsets_y)

    # Generate signal from output resonances
    x_bias = sum(spec_x.filter(lambda _, res: abs(res.frequency) < 1e-3).d)  # remove DC bias
    x_recon = spec_x.reconstruction - x_bias
    y_bias = sum(spec_y.filter(lambda _, res: abs(res.frequency) < 1e-3).d)  # remove DC bias
    y_recon = spec_y.reconstruction - y_bias
    x_signal, y_signal = x_recon, y_recon

    # _, ax = plt.subplots()
    # ax.plot(input_signal, color="tab:blue", alpha=0.5)
    # ax.twinx().twiny().plot(x_signal, y_signal, color="tab:orange", alpha=0.5)
    # plt.show()

    # Compare parametric reconstruction
    _, axs = plt.subplots(3, 3)
    axs[0][0].plot(input_signal.real, color="tab:blue", label="Input")
    axs[0][0].twinx().twiny().plot(x_signal.real, y_signal.real, color="tab:orange", label="Output")
    axs[1][0].plot(input_signal.imag, color="tab:blue", label="Input")
    axs[1][0].twinx().twiny().plot(x_signal.real, y_signal.imag, color="tab:orange", label="Output")
    axs[2][0].plot(abs(input_signal), color="tab:blue", label="Input")
    axs[2][0].twinx().twiny().plot(x_signal.real, abs(y_signal), color="tab:orange", label="Output")
    axs[0][1].plot(input_signal.real, color="tab:blue", label="Input")
    axs[0][1].twinx().twiny().plot(x_signal.imag, y_signal.real, color="tab:orange", label="Output")
    axs[1][1].plot(input_signal.imag, color="tab:blue", label="Input")
    axs[1][1].twinx().twiny().plot(x_signal.imag, y_signal.imag, color="tab:orange", label="Output")
    axs[2][1].plot(abs(input_signal), color="tab:blue", label="Input")
    axs[2][1].twinx().twiny().plot(x_signal.imag, abs(y_signal), color="tab:orange", label="Output")
    axs[0][2].plot(input_signal.real, color="tab:blue", label="Input")
    axs[0][2].twinx().twiny().plot(abs(x_signal), y_signal.real, color="tab:orange", label="Output")
    axs[1][2].plot(input_signal.imag, color="tab:blue", label="Input")
    axs[1][2].twinx().twiny().plot(abs(x_signal), y_signal.imag, color="tab:orange", label="Output")
    axs[2][2].plot(abs(input_signal), color="tab:blue", label="Input")
    axs[2][2].twinx().twiny().plot(abs(x_signal), abs(y_signal), color="tab:orange", label="Output")
    plt.show()
