import numpy as np
from numpy.polynomial.polynomial import Polynomial
from fpt import FptConfig, drs
import matplotlib.pyplot as plt
from serialization import from_file


def transfer(signal_left, signal_right, spectrum_left, spectrum_right, config):
    poly_left = Polynomial(signal_left)  # should be polynomial in 1/z not z
    poly_right = Polynomial(signal_right)
    map = {}
    for z in spectrum_right.z:
        rz = poly_right(z) / poly_left(z)
        for res in spectrum_left.elements:
            input = res.shift(res.max_duration)
            weight_z = input.atz(z) * rz
            weight_w = config.sample_rate * 1j * np.log(weight_z)
            map.setdefault(res, []).append(weight_w)
    return {res: np.array(ws) for res, ws in map.items()}


def transfer3(spectrum_left, spectrum_right):
    map = {}
    for r in spectrum_right.elements:
        Gl = np.sum(spectrum_left.map(lambda _, res: res.at(r.w)))
        for l in spectrum_left.elements:
            gl = l.at(r.w)
            v = r.d * gl / Gl
            map.setdefault(l, []).append(v)
    return {res: np.array(ws) for res, ws in map.items()}


# if __name__ == '__main__':
#     source, sample_rate = from_file('nine')
#     N0 = 10000
#     Ns = 2
#     N = 1000
#     signal = source #[N0:N0 + Ns * N]  # [N0:N0 + 2 * N]
#     # signal_left = source[N0:N0 + N]
#     # signal_right = source[N0 + N:N0 + 2 * N]
#     step_size = N // 2
#     time_unit = step_size / sample_rate
#     config = FptConfig(
#         length=N,
#         degree=N // 2,
#         sample_rate=sample_rate,
#         power_threshold=1e-5,
#     )
#     spectrogram = drs(signal, config, step_size)
#     for n in range(0, len(signal), step_size):
#         spectrum_right = spectrogram.filter(lambda onset, res: onset == n + step_size)
#         signal_right = signal[n + step_size:n + 2 * step_size]
#         spectrum_left = spectrogram.filter(lambda onset, res: onset == n)
#         signal_left = signal[n:n + step_size]
#         if len(spectrum_right) > 0 and len(signal_right) > 0:
#             wss = transfer3(spectrum_left, spectrum_right).items()
#             pairs = [(res, spectrum_right.elements[np.argmax(np.absolute(ws))]) for res, ws in wss]
#             for left, right in pairs:
#                 plt.plot([n / sample_rate, (n + step_size) / sample_rate],
#                          [left.frequency, right.frequency],
#                          c='tab:blue')
#     for n in range(0, len(signal), step_size):
#         spectrum_right = spectrogram.filter(lambda onset, res: onset == n + step_size)
#         spectrum_left = spectrogram.filter(lambda onset, res: onset == n)
#         if len(spectrum_right) > 0:
#             wss = transfer3(spectrum_right, spectrum_left).items()
#             pairs = [(res, spectrum_left.elements[np.argmax(np.absolute(ws))]) for res, ws in wss]
#             for right, left in pairs:
#                 plt.plot([n / sample_rate, (n + step_size) / sample_rate],
#                          [left.frequency, right.frequency],
#                          c='tab:orange')
#     plt.ylim(0, 10000)
#     plt.show()

if __name__ == '__main__':
    source, sample_rate = from_file('zero')
    N0 = 10000
    Ns = 2
    N = 1000
    signal = source[N0:N0 + Ns * N]
    signal_left = source[N0:N0 + N]
    signal_right = source[N0 + N:N0 + 2 * N]
    step_size = N
    config = FptConfig(
        length=N,
        degree=N // 2,
        sample_rate=sample_rate,
        power_threshold=0,
    )
    spectrogram = drs(signal, config, step_size)
    spectrum_left = spectrogram.filter(lambda onset, res: onset == 0)
    spectrum_right = spectrogram.filter(lambda onset, res: onset == 0 + N)
    # wss = transfer(signal_left, signal_right, spectrum_left, spectrum_right, config).items()
    wss = transfer3(spectrum_left, spectrum_right).items()
    # wss = transfer2(spectrum_left, spectrum_right).items()

    # res, ws = wss[0]
    for res, ws in list(wss)[-5:-1]:
        # ws_ = 1 / ws.real
        # # ws = ws_.real
        # ws_normed = ws_ / np.max(ws_)
        # ws_display = ws_normed
        # left = (N0, res.frequency)
        # plt.scatter(N0, res.frequency, res.power)
        # rights = [N0 + N] * len(spectrum_right), spectrum_right.frequency
        # plt.scatter(*rights, s=spectrum_right.power, c=spectrum_right.power)
        # cmap = plt.cm.get_cmap('viridis')(ws_display)
        # for c, (rx, ry, w) in enumerate(zip(*rights, ws_display)):
        #     plt.plot([left[0], rx], [left[1], ry], linewidth=w**2, alpha=w**2, c=cmap[c])
        # plt.ylim(0, res.frequency * 1.1)
        # plt.show()

        plt.figure()
        fws = sorted(zip(spectrum_right.frequency, np.absolute(ws), ws.real, ws.imag), key=lambda fw: fw[0])
        frequencies, magnitudes, reals, imaginaries = list(zip(*fws))
        plt.axvline(res.frequency, color='grey', alpha=0.5)
        plt.plot(frequencies, magnitudes, label='|w|')
        # plt.plot(frequencies, reals, label='Re(w)')
        # plt.plot(frequencies, imaginaries, label='Im(w)')
        plt.title(f'{res.frequency}')
        plt.legend()
        plt.show()
