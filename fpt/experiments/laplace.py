import serialization as ser
from fpt import *
from scipy.signal import residuez
from time import perf_counter
import matplotlib.pyplot as plt

if __name__ == '__main__':
    n0 = 12000
    N = 512
    filename = "nine"
    source, sample_rate = ser.from_file(filename, dir="../data/input")
    signal = source[n0:n0 + N]
    # signal = np.arange(0, 1, 1/N)
    # signal = list(range(512))
    # sample_rate = 1
    step_size = N
    config = FptConfig(
        length=N,
        degree=N // 2,
        delay=0,
        sample_rate=sample_rate,
        power_threshold=0,
        decay_threshold=0
    )

    forward_plus = FptPlus(signal, config).resonances
    backward_plus = FptPlus(np.flip(signal), config).resonances

    forward_minus = FptMinus(signal, config).resonances
    backward_minus = FptMinus(np.flip(signal), config).resonances

    plt.figure()
    plt.plot(signal)
    plt.plot(sum([res.reconstruction for res in forward_plus if res.decay < 0])
             + sum([res.mirror.reconstruction for res in backward_plus if res.mirror.decay > 0]))

    # plt.plot(sum([res.reconstruction for res in forward_plus if res.decay < 0]))
    # plt.plot(sum([res.mirror.reconstruction for res in backward_plus if res.mirror.decay > 0]))

    # plt.figure()
    # plt.plot(signal)
    # plt.plot(sum([np.array([res.mirror.d * res.mirror.z**(n) for n in range(512)]) for res in forward_minus if res.mirror.decay > 0])
    #          + sum([np.array([res.d * res.z**(n) for n in range(512)]) for res in backward_minus if res.decay < 0]))
    # plt.plot(sum([res.reconstruction for res in forward_minus if res.decay < 0]))
    # plt.plot(sum([res.mirror.reconstruction for res in backward_minus if res.mirror.decay > 0]))

    # plt.figure()
    # plt.plot(sum([res.reconstruction for res in forward_plus if res.decay < 0]))
    # plt.plot(sum([res.reconstruction for res in forward_minus if res.decay < 0]))

    # plt.figure()
    # plt.plot(sum([res.mirror.reconstruction for res in backward_plus if res.mirror.decay > 0]))
    # plt.plot(sum([res.mirror.reconstruction for res in backward_minus if res.mirror.decay > 0]))

    # f_p = np.array([res for res in forward_plus if res.decay < 0])
    # f_m = np.array([res for res in forward_minus if res.decay < 0])
    # b_p = np.array([res.mirror for res in backward_plus if res.mirror.decay > 0])
    # b_m = np.array([res.mirror for res in backward_minus if res.mirror.decay > 0])

    # f_pm = []
    # for p in f_p:
    #     for m in f_m:
    #         if abs(m.z - p.z) < 1e-3:
    #             f_pm.append(p)
    #             break
    # b_pm = []
    # for p in b_p:
    #     for m in b_m:
    #         if abs(m.z - p.z) < 1e-3:
    #             b_pm.append(p)
    #             break

    # plt.figure()
    # plt.plot(signal)
    # plt.plot(sum([res.reconstruction for res in f_pm])
    #          + sum([res.reconstruction for res in b_pm]))
    # plt.plot(sum([res.reconstruction for res in f_pm]))
    # plt.plot(sum([res.reconstruction for res in b_pm]))

    # plt.figure()
    # plt.scatter([res.frequency for res in forward_plus if res.decay < 0],
    #             [res.decay for res in forward_plus if res.decay < 0],
    #             alpha=0.5)
    # plt.scatter([res.frequency for res in backward_minus if res.decay < 0],
    #             [res.decay for res in backward_minus if res.decay < 0],
    #             alpha=0.5)
    # plt.scatter([res.frequency for res in f_pm],
    #             [res.decay for res in f_pm],
    #             alpha=0.5,
    #             marker='x')

    # plt.figure()
    # plt.scatter([res.mirror.frequency for res in backward_plus if res.mirror.decay > 0],
    #             [res.mirror.decay for res in backward_plus if res.mirror.decay > 0],
    #             alpha=0.5)
    # plt.scatter([res.mirror.frequency for res in forward_minus if res.mirror.decay > 0],
    #             [res.mirror.decay for res in forward_minus if res.mirror.decay > 0],
    #             alpha=0.5)
    # plt.scatter([res.frequency for res in b_pm],
    #             [res.decay for res in b_pm],
    #             alpha=0.5,
    #             marker='x')

    # f = FptPlus(signal, config)
    # qs = f.qs
    # ps = f.ps
    # zs = f.zs
    # ds = f.ds
    # ds_inv, zs_inv, _ = residuez(ps, qs)
    # ds_, zs_ = ds_inv / zs_inv, 1 / zs_inv
    # mse_ds = sum((ds - ds_) ** 2) / len(ds)
    # mse_zs = sum((zs - zs_) ** 2) / len(zs)
