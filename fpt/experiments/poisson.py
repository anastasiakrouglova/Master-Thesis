from fpt import from_file, drs, FptConfig
import numpy as np
from numpy import fft
import matplotlib.pyplot as plt
from visualization import power_spectrum

N0 = 5000
N = 1024
K = N // 2
inc = 1
signal, sample_rate = from_file("nine")
cs = signal[N0:N0 + N]
ts = np.arange(N) / sample_rate

config = FptConfig(
    length=N,
    degree=K,
    delay=0,
    sample_rate=sample_rate,
    power_threshold=1e-6
)
# 1.0853513244938953
# 0.20505555571337133

# spectrum = drs(cs, config, N)[0]
# power_spectrum(spectrum)

# plt.figure()
# plt.plot(cs)
# plt.show()
for a in [0,100,1000,10000]:
    window = np.exp(-a * np.arange(N)/sample_rate)
    windowed = cs * window
    fig, ax = plt.subplots(2)
    ax0_right = ax[0].twinx()
    ax[0].plot(ts, cs, label="Original", color="tab:blue")
    ax0_right.plot(ts, window, label="Window", color="tab:green")
    ax[0].plot(ts, windowed, label="Windowed", color="tab:orange")
    ax[0].legend()
    ax[0].set_xlabel("Time (s)")
    amps = fft.fftshift(fft.fft(windowed))
    freqs = fft.fftshift(fft.fftfreq(N, 1 / sample_rate))
    ax[1].plot(freqs, amps.real, label="Real")
    ax[1].plot(freqs, amps.imag, label="Imag")
    ax[1].plot(freqs, np.absolute(amps), label="Abs")
    ax[1].set_xlim(-1000, 1000)
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].legend()
    plt.title(f"Exponential window: a={-a}")
    plt.show()
