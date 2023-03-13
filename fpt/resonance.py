from typing import Union, Dict

import numpy as np
from dataclasses import dataclass, replace
from cached_property import cached_property


@dataclass(frozen=True)
class Resonance:
    """The resonance is the basic component of a resonance spectrum.

    In the time domain, it takes the form of a complex damped oscillator.
    In the frequency domain, it is a spectral peak.

    Parameters
    ----------
    d: complex
        The complex amplitude of the resonance.  This corresponds to the
        initial amplitude and phase of the oscillator in the time domain
        and the residue of the spectral peak in the frequency domain.
    w: complex
        The complex angular frequency of the resonance.  This corresponds to the
        angular frequency and decay of the oscillator in the time domain
        and the location of the spectral peak in the frequency domain.
    z: complex
        The complex harmonic variable z = exp(-1j * w * N / sample_rate).
        This is a more useful form for calculations than the corresponding w.
    N: int
        The number of samples over which this resonance is valid. The total
        time duration can be found from N / sample_rate.
    sample_rate: int
        The number of samples per second of the source signal.
        In the literature, sample_rate = 1/tau
    """
    d: complex
    w: complex
    z: complex
    N: int
    sample_rate: int

    # def __post_init__(self):
    #     store.register({hash(self): self})

    def __eq__(self, other):
        return np.allclose([self.d, self.w, self.z, self.N, self.sample_rate],
                           # Test
                           [other.d, other.w, other.z, other.N, other.sample_rate])

    def __lt__(self, other):
        return self.frequency < other.frequency

    def __add__(self, other):
        """Add two resonances.

        Parameters
        ----------
        other: Resonance
            Other resonance to be added to this resonance.

        Returns
        -------
        2-tuple of resonances
            If the resonances have different w, they cannot be combined and
            returned as a pair.
            If the resonance have the same w, the amplitudes are added,
            and a pair is returned with a second element of None.
        """
        if self.w == other.w:
            return replace(self, d=self.d + other.d),
        else:
            return self, other

    def __sub__(self, other):
        """Subtract two resonances

        Parameters
        ----------
        other: Resonance
            Other resonance to be subtracted from this resonance.

        Returns
        -------
        2-tuple of resonances
            If the resonances have different w, they cannot be combined and
            returned as a pair.
            If the resonance have the same w, the amplitudes are subtracted,
            and a pair is returned with a second element of None.
        """
        if self.w == other.w:
            return replace(self, d=self.d - other.d),
        else:
            return self, other

    def __mul__(self, other):
        """Multiply two resonances.

        Notes
        -----
        Assumes all poles are distinct so that the product is still simple.

        Parameters
        ----------
        other: Resonance
            Other resonance to be multiplied with this resonance.

        Returns
        -------
        2-tuple of resonances

        Raises
        ------
        AssertionError
            The two resonances must have different w.
        """
        assert not self.w == other.w
        d_self = (self.d * other.d) / (self.w - other.w)
        d_other = (self.d * other.d) / (other.w - self.w)
        return replace(self, d=d_self), replace(other, d=d_other)

    # def __matmul__(self, other):
    #     if self.decay * other.decay > 0:
    #         ip = self.d * other.d.conj() / (self.w - other.w.conj())
    #         return 2 * np.pi * 1j * np.sign(self.decay) * ip
    #     else:
    #         return 0

    def __matmul__(self, other):
        f0 = 2 * np.pi * self.sample_rate
        left = 2 * 1j * self.d * other.d.conj() / (self.w - other.w.conj())
        right = np.arctan(1j * f0 / self.w) + np.arctan(-1j * f0 / other.w.conj())
        return left * right

    @cached_property
    def norm(self):
        return np.sqrt(self @ self)

    def harmonic(self, h: int):
        amplitude = (1 / h) * self.amplitude  # or 1/1
        phase = - np.sign(self.frequency) * h * self.phase
        d = amplitude * np.exp(1j * phase)
        frequency = h * self.frequency
        decay = self.decay
        w = 2 * np.pi * (frequency + 1j * decay)
        z = np.exp(-1j * w / self.sample_rate)
        return replace(self, d=d, w=w, z=z)

    def harmonics(self, h_max: int):
        return np.array([self.harmonic(h + 1) for h in range(h_max)])

    @cached_property
    def conjugate(self):
        """Resonance: Complex conjugate of the resonance."""
        d_ = self.d.conjugate()
        w_ = self.w.conjugate()
        z_ = self.z.conjugate()
        return replace(self, d=d_, w=w_, z=z_)

    def bin(self, f0: float, f1: float):
        scale = self.amplitude ** 2 / self.decay
        spread = np.angle(self.w - f1) - np.angle(self.w - f0)
        return scale * spread

    @cached_property
    def height(self):
        """float: Height of power resonance peak"""
        return self.amplitude ** 2 / self.decay ** 2

    @cached_property
    def width(self):
        """float: Half width at half height of power resonance peak"""
        return np.abs(self.decay)

    @cached_property
    def area(self):
        """float: Area under the curve of the power resonance peak"""
        return np.pi * self.width * self.height

    @cached_property
    def power(self):
        """float: Average power over N samples"""
        # Fast version is numerically unstable, resulting in NaNs
        # d2, z2, N = self.amplitude ** 2, abs(self.z) ** 2, self.N
        # return (d2 / N) * ((z2 ** N - 1) / (z2 - 1))
        return np.sum(np.absolute(self.reconstruction) ** 2) / self.N

    @cached_property
    def frequency(self):
        """float: Frequency in Hz"""
        return self.w.real / (2 * np.pi)

    @cached_property
    def decay(self):
        """float: Decay in Hz"""
        return self.w.imag / (2 * np.pi)

    @cached_property
    def amplitude(self):
        """float: Initial amplitude"""
        return abs(self.d)

    @cached_property
    def phase(self):
        """float: Phase in radians (-pi, pi)"""
        return np.angle(self.d)

    @cached_property
    def reconstruction(self):
        """[complex]: Reconstructed time signal"""
        if self.amplitude == 0:  # < 1e-30:
            return np.zeros(self.N)
        else:
            return self.d * np.power(self.z, np.arange(0, self.N))

    @cached_property
    def mirror(self):
        d_ = self.d * self.z ** (self.N - 1)
        w_ = -self.w
        z_ = 1 / self.z
        return replace(self, d=d_, w=w_, z=z_)

    def at(self, w: complex):
        """Evaluate this resonance at point w.

        Parameters
        ----------
        w: complex
            Point at which to evaluate the resonance.
            Normally, this is a real number but can be complex in general.

        Returns
        -------
        complex
            Value of resonance at given point.
        """
        return self.d / (w - self.w)

    def atz(self, z: complex):
        """Evaluate this resonance at point z.

        Parameters
        ----------
        z: complex
            Point at which to evaluate the resonance.
            Normally, this is a real number but can be complex in general.

        Returns
        ------- 2.18140452e-05+3.15084234e-12j(1-3.10797479e-09j)
        complex
            Value of resonance at given point.
        """
        return self.d / (z - self.z)

    def shift(self, offset: int):
        """Shift the resonance forward in time by altering the amplitude d.

        Parameters
        ----------
        offset: int
            Number of samples to shift the resonance forward.

        Returns
        -------
        Resonance
            This resonance with shifted amplitude.
        """
        return replace(self, d=self.d * self.z ** offset)

    def truncate(self, offset: int):
        """Truncate the number of samples over which this resonance is valid.

        Parameters
        ----------
        offset: int
            Number of samples that the resonance will be valid for.

        Returns
        -------
        Resonance
            This resonance with truncated duration.
        """
        assert offset <= self.N
        return replace(self, N=offset)

    def resample(self, scale: float):
        """Resample the resonance by scaling the sample rate and valid samples.

        Parameters
        ----------
        scale: float
            Factor by which the sample rate and duration are scaled.

        Returns
        -------
        Resonance
            This resonance with scaled duration and sample rate.
        """
        N_ = int(self.N * scale)
        sample_rate_ = int(self.N * scale)
        return replace(self, N=N_, sample_rate=sample_rate_)

    def overlap(self, other):
        """Amount of spectral overlap between two resonances.

        Parameters
        ----------
        other: Resonance
            Other resonance with which to evaluate the overlap.

        Returns
        -------
        float
            Amount of overlap between two resonances.
        """
        assert not self.w == other.w
        return ((self.amplitude * other.amplitude) / abs(self.w - other.w)) ** 2

    def attenuate(self, scale: float):
        return replace(self, d=self.d * scale)


# Type Aliases
# ID = int
Onset = int


@dataclass
class ResonanceSet:
    elements: [Resonance]
    onsets: [Onset]

    # def __post_init__(self):
    #     idxs = np.argsort(self.elements)
    #     self.elements = self.elements[idxs]
    #     self.onsets = self.onsets[idxs]

    # @cached_property
    # def elements(self):
    #     els = np.array([store[el] for el in self._elements])
    #     return els

    # def __hash__(self):
    #     return hash((self._elements.tobytes(), self.onsets.tobytes()))

    def __len__(self):
        return len(self.elements)

    def __add__(self, other):
        # Assume disjoint?
        return ResonanceSet(
            np.concatenate((self.elements, other.elements)),
            np.concatenate((self.onsets, other.onsets)))

    def __sub__(self, other):
        eos = [(res, onset) for res, onset
               in zip(self.elements, self.onsets)
               if res not in other.elements]
        if len(eos) > 0:
            elements, onsets = zip(*eos)
            return ResonanceSet(np.array(elements), np.array(onsets))
        else:
            return ResonanceSet(np.array([]), np.array([]))

    def __getitem__(self, subscript):
        return ResonanceSet(self.elements[subscript], self.onsets[subscript])

    def __getattr__(self, item):
        if isinstance(item, str) and item[:2] == item[-2:] == '__':
            raise AttributeError(item)
        else:
            return self.map(item)

    def __matmul__(self, other):
        da, wa, fa = self.d.reshape(-1, 1), self.w.reshape(-1, 1), self.sample_rate.reshape(-1, 1)
        db, wb, fb = other.d.reshape(1, -1), other.w.reshape(1, -1), other.sample_rate.reshape(1, -1)
        ds = da * db.conjugate()
        ws = wa - wb.conjugate()
        ts = np.arctan(2 * np.pi * 1j * fa / wa) + np.arctan(-1j * 2 * np.pi * fb / wb.conjugate())
        return np.sum(2 * 1j * ds * ts / ws)

    # def __matmul__(self, other):
    #     return sum([left @ right
    #                 for left in self.elements
    #                 for right in other.elements])

    @cached_property
    def norm(self):
        n = np.sqrt(self @ self)
        assert (np.isnan(n.imag) or np.isclose(n.imag, 0))
        return n.real

    def harmonic(self, h: int):
        return self.transform(lambda _, res: res.harmonic(h))

    def harmonics(self, h_max: int):
        return sum([self.harmonic(h + 1) for h in range(h_max)],
                   start=ResonanceSet(np.array([]), np.array([])))

    def map(self, fn):
        # fn is a property
        if isinstance(fn, str):
            return np.array([getattr(el, fn)
                             for el in self.elements])
        # fn is a function
        else:
            return np.array([fn(onset, el)
                             for onset, el in zip(self.onsets, self.elements)])

    def map_with(self, fn, args):
        if isinstance(fn, str):
            return np.array([getattr(el, fn)(arg)
                             for el, arg in zip(self.elements, args)])
        else:
            return np.array([fn(onset, el, arg)
                             for onset, el, arg in zip(self.onsets, self.elements, args)])

    def transform(self, fn):
        return replace(self, elements=self.map(fn))

    def transform_with(self, fn, args):
        return replace(self, elements=self.map_with(f, args))

    def filter(self, pred):
        mask = self.map(pred)
        return replace(self, elements=self.elements[mask], onsets=self.onsets[mask])

    def filter_with(self, pred, args):
        mask = self.map_with(pred, args)
        return replace(self, elements=self.elements[mask], onsets=self.onsets[mask])

# # NET TOEGEVOEGD
#     @cached_property
#     def N(self):
#         return np.max([el.max_duration + onset for onset, el in zip(self.onsets, self.elements)])

    # @cached_property
    # def reconstruction(self):
    #     recon = np.zeros(self.N, dtype=np.complex128)
    #     for onset, el in zip(self.onsets, self.elements):
    #         recon[onset:onset + el.N] += el.reconstruction
    #     return recon

# @dataclass
# class ResonanceSpectrum(ResonanceSet):
#     pass


# @dataclass
# class ResonanceSpectrogram(ResonanceSet):
#     pass

# @dataclass
# class ResonanceStore:
#     store: Dict[ID, Union[Resonance, ResonanceSet]]
#
#     def __getitem__(self, item):
#         return self.store[item]
#
#     def register(self, other):
#         if isinstance(other, tuple):
#             self.store.update({other[0]: other[1]})
#         elif isinstance(other, dict):
#             self.store.update(other)
#         else:
#             raise ValueError
#
#     @property
#     def ids(self):
#         return np.array(list(self.store.keys()))
#
#     @property
#     def elements(self):
#         return np.array(list(self.store.values()))
#
#
# Module-level resonance store
# store = ResonanceStore({})
