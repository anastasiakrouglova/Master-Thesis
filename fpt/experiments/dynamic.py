from dataclasses import dataclass, replace
from cached_property import cached_property
from resonance import Resonance
import numpy as np


@dataclass(frozen=True)
class DynamicResonance:
    sequence: [Resonance]
    onset: int

    @cached_property
    def reconstruction(self):
        # Assumes resonances do not overlap in time
        return np.concatenate([resonance.reconstruction
                               for resonance in self.sequence])


def horizontalize(spectrogram, index_maps):
    dynamic_resonances = []
    previous = {}
    for index_map, onset, left_spectrum, right_spectrum in \
            zip(index_maps, spectrogram.onsets,
                spectrogram.spectra, spectrogram.spectra[1:], spectrogram.onsets):
        current = {}

        # Extend (or start) any dynamic resonances in previous
        for left_index, right_index in index_map:
            if left_index in previous:
                dyn = previous.pop(left_index)
            else:
                dyn = DynamicResonance([left_spectrum[left_index]], onset)
            sequence = dyn.sequence.append(right_spectrum[right_index])
            current[right_index] = replace(dyn, sequence=sequence)

        # Any dynamic resonances left in previous are finished
        for dyn in previous.values():
            dynamic_resonances.append(dyn)
        previous = current.copy()
    return dynamic_resonances
