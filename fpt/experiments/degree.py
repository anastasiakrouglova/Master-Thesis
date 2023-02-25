import numpy as np
import matplotlib.pyplot as plt
from fpt import from_file
import mpmath

mpmath.mp.dps = 200
np.set_printoptions(suppress=True, precision=3)

signal, sample_rate = from_file("nine")
N0 = 10000
N = 2048
M = N
cs = signal[N0:N0+N]

# ns = np.linspace(0, 10 * 2 * np.pi, N)
# cs = np.exp(-0.1*ns) * np.sin(ns)

# plt.plot(cs)
# plt.show()

# def e(n, mp1):
#     m = mp1 - 1
#     # print(n, m)
#     if m == -1:
#         return 0
#     elif m == 0:
#         return cs[n]
#     else:
#         return e(n + 1, m - 1) + 1 / (e(n + 1, m) - e(n, m))

# e = np.zeros((N, M))
e = [[mpmath.mpf(0) for _ in range(M)] for _ in range(N)]
# e[:] = np.nan
#
for n, c in enumerate(cs):
    e[n][0] = mpmath.mpf(c)
# e[:, 0] = [mp.mpf(c) for c in cs]

for m in range(M - 1):
    for n in range(N - m - 1):
        if e[n + 1][m] == e[n][m]:
            e[n][m + 1] = mpmath.inf
        else:
            if m == 0:
                e[n][m + 1] = 1 / (e[n + 1][m] - e[n][m])
            else:
                e[n][m + 1] = e[n + 1][m - 1] + 1 / (e[n + 1][m] - e[n][m])

logabs = np.log(np.absolute(np.array(e[0][::2], dtype=float)))
plt.plot(logabs)
plt.axhline(color='grey', alpha=0.5)
plt.axvline(np.argmin(logabs), color='grey', alpha=0.5)
plt.show()

# evens = e[:, ::2][:,1:]
# min = np.unravel_index(np.nanargmin(np.absolute(evens)), evens.shape)
# print(min, evens[min])
# print(evens[0,-1])


# plt.contourf(evens,levels=1)
# plt.ylabel("N")
# plt.xlabel("M")
# plt.colorbar()
# plt.show()
