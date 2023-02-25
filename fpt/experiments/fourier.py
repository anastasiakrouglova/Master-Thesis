import sys
sys.path.insert(0, '/Users/nastysushi/Mirror/_MULTIMEDIA/THESIS/resonances/fpt-master')

import resonance
import numpy as np
from dataclasses import dataclass, replace
from cached_property import cached_property


@dataclass(frozen=True)
class Resonance(resonance.Resonance):
    @cached_property
    def bounded(self):
        """Resonance: Bounded resonance stops after N samples."""
        return (self - self.shift(self.N))[0]


@dataclass(frozen=True)
class ResonanceSpectrum(resonance.ResonanceSpectrum):

    @cached_property
    def bounded(self):
        """ResonanceSpectrum: Bounded spectrum stops after N samples."""
        boundeds = np.array([resonance.bounded
                             for resonance in self.spectrum])
        return replace(self, spectrum=boundeds)

    @cached_property
    def fourier(self):
        """[complex]: Discrete Fourier coefficients of this spectrum."""
        grid = [-2 * np.pi * self.sample_rate * n / self.max_duration
                for n in range(-self.max_duration // 2, self.max_duration // 2)]
        return 1j * self.sample_rate * np.array([self.bounded.at(val)
                                                 for val in grid])
