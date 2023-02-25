import numpy as np
import numpy.linalg as la
from dataclasses import dataclass
from cached_property import cached_property
from abc import ABC, abstractmethod
from numpy.polynomial.polynomial import polyroots
from enum import Flag, auto
from resonance import Resonance, ResonanceSet


class FptMode(Flag):
    Plus = auto()
    Minus = auto()
    Both = auto()


@dataclass(frozen=True)
class FptConfig:
    length: int
    degree: int
    delay: int = 0
    sample_rate: int = 44100
    convergence_threshold: float = None
    amplitude_threshold: float = None
    power_threshold: float = None
    decay_threshold: float = None
    mode: FptMode = FptMode.Both


def drs(signal: np.ndarray, config: FptConfig, step_size: int):
    return StFpt(signal, config, step_size).spectrogram


@dataclass(frozen=True)
class StFpt:
    signal: np.ndarray
    config: FptConfig
    step_size: int = None

    @cached_property
    def padded(self):
        N = len(self.signal)
        W = self.config.length
        S = self.step_size
        left_pad_length = (W - S) // 2
        left_pad = np.zeros(left_pad_length)
        right_pad_length = S - (N % S) + left_pad_length
        right_pad = np.zeros(right_pad_length)
        return np.concatenate((left_pad, self.signal, right_pad))

    @cached_property
    def onsets(self):
        return np.array([n for n in range(0, len(self.signal), self.step_size)])

    @cached_property
    def slices(self):
        return np.array([self.padded[n:n + self.config.length] for n in self.onsets])

    @cached_property
    def spectrogram(self):
        N = len(self.signal)
        offset = (self.config.length - self.step_size) // 2
        truncations = [self.step_size] * (len(self.onsets) - 1) + [N - self.onsets[-1]]
        resonances = []
        onsets = []
        for onset, slice, truncation in zip(self.onsets, self.slices, truncations):
            combined = bifpt(slice, self.config, offset, truncation)
            resonances += list(combined)
            # ids += [hash(res) for res in combined]
            onsets += [onset] * len(combined)
            print(f"\rProgress: {onset / N:.2%}")
        spectrogram = ResonanceSet(np.array(resonances), np.array(onsets))
        # store.register({hash(spectrogram): spectrogram})
        return spectrogram


def bifpt(cs: np.ndarray, config: FptConfig, offset: int = 0, truncation: int = None):
    if truncation is None:
        truncation = len(cs)

    inside = FptPlus(cs, config).resonances
    inside_shifted = np.array([res.shift(offset).truncate(truncation) for res in inside])
    forward = inside_shifted[[res.decay < 0 and res.power > config.power_threshold for res in inside_shifted]]

    outside = FptPlus(np.flip(cs), config).resonances
    outside_shifted = np.array([res.mirror.shift(offset).truncate(truncation) for res in outside])
    reverse = outside_shifted[[res.decay > 0 and res.power > config.power_threshold for res in outside_shifted]]

    combined = np.concatenate((forward, reverse))
    return combined


@dataclass(frozen=True)
class Fpt(ABC):
    cs: np.ndarray
    config: FptConfig

    @cached_property
    def resonances(self):
        N = len(self.cs)
        sr = self.config.sample_rate
        return np.array([Resonance(d, w, z, N, sr)
                         for d, w, z in zip(self.ds, self.ws, self.zs)])

    @abstractmethod
    @cached_property
    def qs(self):
        ...

    @abstractmethod
    @cached_property
    def ps(self):
        ...

    @abstractmethod
    @cached_property
    def zs(self):
        ...

    @abstractmethod
    @cached_property
    def ds(self):
        ...

    @abstractmethod
    @cached_property
    def ws(self):
        ...


@dataclass(frozen=True)
class FptMinus(Fpt):

    @cached_property
    def qs(self):
        N = len(self.cs)
        K = min(self.config.degree, N // 2)
        s = self.config.delay
        M = N - 1 - K - s
        A = np.array([np.flip(self.cs[s + 1 + i:K + s + 1 + i])
                      for i in range(M)])
        b = - np.array(self.cs[K + s + 1:K + s + M + 1])
        q = la.lstsq(A, b, rcond=None)[0]
        return np.insert(q, 0, 1)

    @cached_property
    def ps(self):
        assert self.qs is not None
        s = self.config.delay
        K = min(self.config.degree, len(self.cs) // 2)
        A = np.array([np.append(np.flip(self.cs[s:i + s + 1]), [0] * (K - i))
                      for i in range(K + 1)])
        return A @ self.qs

    @cached_property
    def zs(self):
        assert self.qs is not None
        # return polyroots(np.flip(self.qs))
        return 1 / polyroots(self.qs)

    @cached_property
    def ds(self):
        assert self.zs is not None
        K = min(len(self.zs), len(self.cs) // 2)
        i = np.arange(K + 1)
        exponent = -np.tile(i, (K, 1))

        if len(self.ps) > K + 1 or len(self.qs) > K + 1:
            ps = self.ps[:K + 1]
            qs = self.qs[:K + 1]
            # ps = np.trim_zeros(self.ps, trim='b')
            # qs = np.trim_zeros(self.qs, trim='b')
        else:
            ps = self.ps
            qs = self.qs

        Z = np.power(self.zs.reshape(-1, 1), exponent)
        d = np.divide(Z @ self.ps.reshape(-1, 1),
                      Z @ (-self.qs * i).reshape(-1, 1))
        return d.reshape((-1,))

    @cached_property
    def ws(self):
        assert self.zs is not None
        return self.config.sample_rate * 1j * np.log(self.zs)


@dataclass(frozen=True)
class FptPlus(Fpt):

    @cached_property
    def qs(self):
        N = len(self.cs)
        K = min(self.config.degree, N // 2)
        s = self.config.delay
        M = N - 1 - K - s
        A = np.array([self.cs[s + 1 + i:s + K + 1 + i]
                      for i in range(M + 1)])
        b = - np.array(self.cs[s:s + M + 1])
        q = la.lstsq(A, b, rcond=None)[0]
        return np.insert(q, 0, 1)

    @cached_property
    def ps(self):
        assert self.qs is not None
        K = min(self.config.degree, len(self.cs) // 2)
        s = self.config.delay
        A = np.array([np.append([0] * i, self.cs[s:s + K - i])
                      for i in range(K)])
        p = A @ self.qs[1:]
        return np.insert(p, 0, 0)

    @cached_property
    def zs(self):
        assert self.qs is not None
        return polyroots(self.qs)

    @cached_property
    def ds(self):
        assert self.zs is not None
        K = min(len(self.zs), len(self.cs) // 2)
        i = np.arange(1, K + 1)
        exponent = np.tile(i, (K, 1))
        Z = np.power(self.zs.reshape(-1, 1), exponent)
        # Not necessary in FPT-?
        if len(self.ps) > K + 1 or len(self.qs) > K + 1:
            ps = self.ps[:K + 1]
            qs = self.qs[:K + 1]
            # ps = np.trim_zeros(self.ps, trim='b')
            # qs = np.trim_zeros(self.qs, trim='b')
        else:
            ps = self.ps
            qs = self.qs
        d = np.divide(Z @ ps[1:].reshape(-1, 1),
                      Z @ (qs[1:] * i).reshape(-1, 1))
        return d.reshape((-1,))

    @cached_property
    def ws(self):
        assert self.zs is not None
        return self.config.sample_rate * 1j * np.log(self.zs)
