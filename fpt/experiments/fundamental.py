import fpt
from fpt import FptConfig, drs, store
from visualization import discrete
import serialization as ser
from matplotlib import pyplot as plt
from time import perf_counter
from numpy.polynomial.polynomial import polyroots
import numpy as np

if __name__ == '__main__':
    n0 = 0
    N = 500
    filename = 'tester'
    source, sample_rate = ser.from_file(filename)
    signal = source  # [n0:n0 + N]
    step_size = N  # // 2
    config = FptConfig(
        length=N,
        degree=N // 2,
        delay=0,
        sample_rate=sample_rate,
        power_threshold=1e-8,
        decay_threshold=0
    )

    inside = fpt.FptPlus(signal, config)
    zqs_in_all = polyroots(inside.qs)
    zqs_in = zqs_in_all[np.bitwise_and(np.absolute(zqs_in_all) < 1, np.absolute(zqs_in_all) > 0)]
    zps_in_all = polyroots(inside.ps)
    zps_in = zps_in_all[np.bitwise_and(np.absolute(zps_in_all) < 1, np.absolute(zps_in_all) > 0)]

    outside = fpt.FptPlus(np.flip(signal), config)
    zqs_out_all = polyroots(outside.qs)
    zqs_out = zqs_out_all[np.bitwise_and(np.absolute(zqs_out_all) < 1, np.absolute(zqs_out_all) > 0)]
    zps_out_all = polyroots(outside.ps)
    zps_out = zps_out_all[np.bitwise_and(np.absolute(zps_out_all) < 1, np.absolute(zps_out_all) > 0)]

    fs = np.linspace(1, 22050, 20000)
    sum_zqs_in = np.array([np.sum(f * zqs_in ** (sample_rate / f)) for f in fs])
    sum_zps_in = np.array([np.sum(f * zps_in ** (sample_rate / f)) for f in fs])
    combined_in = sum_zps_in - sum_zqs_in

    sum_zqs_out = np.array([np.sum(f * zqs_out ** (sample_rate / f)) for f in fs])
    sum_zps_out = np.array([np.sum(f * zps_out ** (sample_rate / f)) for f in fs])
    combined_out = sum_zps_out - sum_zqs_out

    combined = combined_in + combined_out

    plt.plot(fs, sum_zqs_in, color="tab:blue", linestyle="dashed", alpha=0.5)
    plt.plot(fs, sum_zps_in, color="tab:blue", linestyle="dotted", alpha=0.5)
    plt.plot(fs, combined_in, color="tab:blue", alpha=0.5)
    plt.plot(fs, sum_zqs_out, color="tab:orange", linestyle="dashed", alpha=0.5)
    plt.plot(fs, sum_zps_out, color="tab:orange", linestyle="dotted", alpha=0.5)
    plt.plot(fs, combined_out, color="tab:orange", alpha=0.5)
    plt.plot(fs, combined, color="tab:green")
    plt.show()
