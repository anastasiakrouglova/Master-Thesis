# Insert path as long as this file is inside the notebooks folder.

import sys

sys.path.insert(0, '../')

from fpt import *

import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from scipy import signal as spsignal
from scipy.io.wavfile import read, write
from scipy.special import voigt_profile as voigt
import IPython.display as ipd
from resonance import ResonanceSet

"""
Helper for Resonance and ResonanceSet operations

Creation of resonance set:
    res_set = ResonanceSet(np.array([spectrogram.elements[0]]), np.array(spectrogram.onsets[0]))

Creation of set of sets of resonances: 
    res_res_set = ResonanceSet(np.array([hash(res_set), hash(res_set)]), np.array(onsets))

Filter spectrogram:
    spectrogram.filter(lambda onset, res: res.frequency < 2000)

Filter spectrogram with external attribute:
    spectrogram.filter_with(lambda onset, res, frequency: frequency > 0, spectrogram.frequency)

"""



def get_spectrogram(file, folder=None, N=500, step_size=500, rem_offset=False, t_min=0, t_max=0,
                    amp_threshold=1e-8, power_threshold=1e-9):
    """Function to retrieve the FPT spectrogram of a signal given the filename of the signal

    Parameters
    ----------
    file : str
        The filename of the time signal inside /data folder
    folder : str, optional
        Specific folder or folder path inside /data folder where file is located (default is None)
    N : int, optional
        Window length of the FPT (default is 500)
    step_size : int, optional
        Step size of the FPT (default is 500)
    rem_offset bool, optional
         Remove offset of the time signal (default is False)
    t_min : float, optional
        Start time in seconds of the signal for the FPT (default is 0)
    t_max : float, optional
        End time in seconds of the signal for the FPT. If t_max is 0, the FPT is obtained until 
        the end of the signal (default is 0)
    amp_threshold : float, optional
        Amplitude threshold, resonances with smaller amplitude than the threshold are filtered
        (default is 1e-8)
    power_threshold : float, optional
        Power threshold, resonances with smaller power than the threshold are filtered
        (default is 1e-9)

    Returns
    -------
    resonance3.ResonanceSpectrogram
        Resonance spectrogram of the input time signal
    numpy.ndarray
        Normalized time signal
    """

    # Load signal
    if folder == None:
        #sample_rate, signal = read(filename='./fpt/data/input/' + file)
        #sample_rate, signal = read(filename='./data/input/' + file)
        sample_rate, signal = read(filename='./' + file)
    else:
        sample_rate, signal = read(filename='./data/' + folder + '/' + file)
        #sample_rate, signal = read(filename='./fpt/data/input/' + folder + '/' + file)

    # Convert to mono
    if len(signal.shape) > 1:
        signal = signal[:, 0]

    # Slice signal from t_min to t_max
    if t_max > 0 and t_min > 0:
        signal = signal[int(t_min * sample_rate):int(t_max * sample_rate)]
    elif t_max > 0:
        signal = signal[0:int(t_max * sample_rate)]

    # Normalize
    norm_factor = (np.max(np.abs(signal)))
    signal_norm = signal / norm_factor

    # Remove offset
    if rem_offset:
        signal_norm = signal_norm - np.average(signal_norm)

    # Configure parameter
    config = FptConfig(
        length=N,
        degree=N // 2,
        delay=0,
        sample_rate=sample_rate,
        amplitude_threshold=amp_threshold,
        power_threshold=power_threshold,
        # convergence_threshold=conv_threshold,
        decay_threshold=0
    )

    # Perform short-term fast padé transform

    spectrogram = drs(signal_norm, config, step_size)

    return spectrogram, signal


def plot_spectrogram(spectrogram, signal=None, min_freq=0, max_freq=3000, plot_fourier=False, scale=1):
    """Function to plot the resonance spectrogram scaled by power

    Parameters
    ----------
    spectrogram : resonance3.ResonanceSpectrogram
        Resonance spectrogram to plot
    signal : numpy.ndarray, optional
        Time signal, used when the Fourier Transform is added to the plot (default is None)
    min_freq : int, optional
        Lower bound frequency (Hz) of the plot (default is 0)
    max_freq : int, optional
        Upper bound frequency (Hz) of the plot (default is 3000)
    plot_fourier bool, optional
         Add the Fast Fourier Spectrogram to the plot (default is False)
    scale float, optional
         Scale size of resonances (points of the scatterplot) (default is 1)
    """

    fig = plt.figure(figsize=(15, 8))

    ax = fig.add_subplot(111, label="1")

    # Original spectrogram    
    ax.scatter(spectrogram.onsets / spectrogram.sample_rate, spectrogram.frequency,
               s=np.array(spectrogram.power) * 10 * scale,
               c=np.array(spectrogram.power), zorder=2)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_ylim((min_freq, max_freq))

    # Add Fourier representation:
    if plot_fourier:
        if signal is not None:
            f, t, Sxx = spsignal.spectrogram(signal, spectrogram.sample_rate[0])
            ax.pcolormesh(t, f, np.log(Sxx), shading='gouraud', zorder=1)

        else:
            print("You must pass the original signal to obtain the Fourier representation")

    plt.title('Resonance Spectrogram')


def plot_spectrogram_arrows(spectrogram, min_freq=0, max_freq=3000, scale=1):
    """Function to plot the resonance spectrogram with symbols of the scatter plot corresponding to decay

    Parameters
    ----------
    spectrogram : resonance3.ResonanceSpectrogram
        Resonance spectrogram to plot
    min_freq : int, optional
        Lower bound frequency (Hz) of the plot (default is 0)
    max_freq : int, optional
        Upper bound frequency (Hz) of the plot (default is 3000)
    scale float, optional
         Scale size of resonances (points of the scatterplot) (default is 1)
    """

    fig = plt.figure(figsize=(15, 8))

    ax = fig.add_subplot(111, label="1")

    # Original spectrogram    
    damping = ResonanceSet(spectrogram.elements[np.array(spectrogram.w).imag < - 5],
                           spectrogram.onsets[np.array(spectrogram.w).imag < - 5])

    ramping = ResonanceSet(spectrogram.elements[np.array(spectrogram.w).imag > 5],
                           spectrogram.onsets[np.array(spectrogram.w).imag > 5])

    mask = np.logical_and(np.array(spectrogram.w).imag > -5, np.array(spectrogram.w).imag < 5)
    constant = ResonanceSet(spectrogram.elements[mask], spectrogram.onsets[mask])

    ax.scatter(damping.onsets / damping.sample_rate, damping.frequency, marker="v", label="Damping",
               s=20 + np.array(damping.power) * 10 * scale, c=np.array(damping.power), zorder=2)

    ax.scatter(ramping.onsets / ramping.sample_rate, ramping.frequency, marker="^", label="Ramping",
               s=20 + np.array(ramping.power) * 10 * scale, c=np.array(ramping.power), zorder=2)

    ax.scatter(constant.onsets / constant.sample_rate, constant.frequency, label="'Constant'",
               s=20 + np.array(constant.power) * 10 * scale, c=np.array(constant.power), zorder=2)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_ylim((min_freq, max_freq))

    plt.legend()
    plt.title('Resonance Spectrogram')


""" Dynamic Resonances related funtions """


def residue_distance(res, candidates, forward):
    """Function to calculate the distances from a resonance to each of the candidate resonances
    based on the residue distance.

    Parameters
    ----------
    res : resonance3.Resonance
        Resonance for which we obtain the distances
    candidates : resonance3.ResonanceSet
        Resonance set of the candidate resonances for which we obtain the distances from res
    forward : bool
        True if res is res is rolled forward (res is previous to candidates in time), 
        False if res is res is rolled backwards (candidates are previous to res in time)

    Returns
    -------
    numpy.ndarray
        Array of distances from res to candidates
    """

    # Shift the res resonance
    shifted_res = Resonance(res.d * np.exp((1 if forward else -1) * -1j * res.w * res.N / res.sample_rate), res.w,
                            res.z, res.N, res.sample_rate)

    # Calculate the distances
    return np.abs(shifted_res.d) * np.abs(candidates.d) / np.abs(res.w - candidates.w)


def residue_power_distance(res, candidates, forward):
    """Function to calculate the distances from a resonance to each of the candidate resonances
    based on the residue distance weighted by the power.

    Parameters
    ----------
    res : resonance3.Resonance
        Resonance for which we obtain the distances
    candidates : resonance3.ResonanceSet
        Resonance set of the candidate resonances for which we obtain the distances from res
    forward : bool
        True if res is res is rolled forward (res is previous to candidates in time), 
        False if res is res is rolled backwards (candidates are previous to res in time)

    Returns
    -------
    numpy.ndarray
        Array of distances from res to candidates
    """

    # Shift the res resonance
    shifted_res = Resonance(res.d * np.exp((1 if forward else -1) * -1j * res.w * res.N / res.sample_rate), res.w,
                            res.z, res.N, res.sample_rate)

    # Calculate the distances
    return res.power * np.array(candidates.power) * np.abs(shifted_res.d) * np.abs(candidates.d) / np.abs(
        res.w - candidates.w)


def spectral_distance(res, candidates):
    """Function to calculate the distances from a resonance to each of the candidate resonances
    based on the residue spectral distance (frequency distance).

    Parameters
    ----------
    res : resonance3.Resonance
        Resonance for which we obtain the distances
    candidates : resonance3.ResonanceSet
        Resonance set of the candidate resonances for which we obtain the distances from res

    Returns
    -------
    numpy.ndarray
        Array of distances from res to candidates
    """

    # Calculate the distances
    return 1 / np.abs(res.w - candidates.w)


def harmonic_mean_distance(res, candidates, forward, alpha=100, beta=1):
    """Function to calculate the distances from a resonance to each of the candidate resonances
    based on the harmonic mean of the w and d attributes.

    Parameters
    ----------
    res : resonance3.Resonance
        Resonance for which we obtain the distances
    candidates : resonance3.ResonanceSet
        Resonance set of the candidate resonances for which we obtain the distances from res
    forward : bool
        True if res is res is rolled forward (res is previous to candidates in time), 
        False if res is res is rolled backwards (candidates are previous to res in time)
    alpha : float, optional
        Regularization parameter for the w (default is 100)
    beta : float, optional
        Regularization parameter for the d (default is 1)

    Returns
    -------
    numpy.ndarray
        Array of distances from res to candidates
    """

    # Calculate the distances
    return 1 / (alpha * np.abs(res.w - candidates.w) / (res.sample_rate * 2 * np.pi) +
                beta * np.abs(
                (res.d * np.exp((1 if forward else -1) * -1j * res.w * res.N / res.sample_rate)) - candidates.d))


def residue_distance_transference(res, candidates, spectrum_l):
    """Function to calculate the distances from a resonance to each of the candidate resonances
    based on the residue distance. res resonance is multiplied by the transference function.

    Parameters
    ----------
    res : resonance3.Resonance
        Resonance for which we obtain the distances
    candidates : resonance3.ResonanceSet
        Resonance set of the candidate resonances for which we obtain the distances from res
    spectrum_l : resonance3.ResonanceSet
        Spectrum of the res resonance.

    Returns
    -------
    numpy.ndarray
        Array of distances from res to candidates
    """

    # Calculate the distances
    return np.abs(
        (res.d * np.array(candidates.d)) / (np.array([np.sum([res_l.d / (w - res_l.w) for res_l in spectrum_l.elements])
                                                      for w in candidates.w]) * (np.array(candidates.w) - res.w)))


def residue_power_distance_transference(res, candidates, spectrum_l):
    """Function to calculate the distances from a resonance to each of the candidate resonances
    based on the residue distance weighted by the power. res resonance is multiplied by the 
    transference function.

    Parameters
    ----------
    res : resonance3.Resonance
        Resonance for which we obtain the distances
    candidates : resonance3.ResonanceSet
        Resonance set of the candidate resonances for which we obtain the distances from res
    spectrum_l : resonance3.ResonanceSet
        Spectrum of the res resonance.

    Returns
    -------
    numpy.ndarray
        Array of distances from res to candidates
    """

    # Calculate the distances
    return res.power * np.array(candidates.power) * np.abs(
        (res.d * np.array(candidates.d)) / (np.array([np.sum([res_l.d / (w - res_l.w)
                                                              for res_l in spectrum_l.elements])
                                                      for w in candidates.w]) * (np.array(candidates.w) - res.w)))


def overlap_function(spectrogram, freq_ratio=5, min_overlap=1e-5, overlap_type="constant", windows=1,
                     multiple_match=False, forward=True, distance="residue_power"):
    """Function to match resonances from contiguous spectra (slices) into dynamic resonances based on a resonance distance.

    Parameters
    ----------
    spectrogram : resonance3.ResonanceSpectrogram # SHOULD BE ResonanceSet 
        Spectrogram for which matches between resonances are found
    freq_ratio : float, optional
        Ratio of frequency in times/second relative to the original resonance for which candidates 
        are considered (default is 5)    
    min_overlap : float, optional
        Min overlap between resonances to consider it as a match (default is 1e-5)
    overlap_type : str, optional
        Rate of change of the min_overlap along the frequency. Options are:
            'constant': constant min_overlap 
            'exponential': min_overlap * 10 ^ (Resonance frequency / 3000)
        (default is 'constant')
    windows : int, optional
        Number of contiguous windows from where candidates are considered (default is 1)
    multiple_match bool, optional
         Allow multiple that multiple resonances can math into a single candidate (default is False)
    forward : bool, optional
        Time direction of the matching (default is True)
    distance : str, optional
        Defined distance type to use. Options are:
            harmonic_mean
            residue
            residue_power
            spectral
            residue_transference
            residue_power_transference
        (default is 'residue_power')

    Returns
    -------
    List[List[Tuple]]
        List of matches between resonances as indexes (Tuple) for each slice (spectrum)
  
    """

    matches_list = []


    # Convert freq ratio from times/second to times/window:
    freq_ratio = 1 + freq_ratio * spectrogram.elements[0].N / spectrogram.sample_rate[0]

    # Obtain list of spectra
    spectrum_onsets = np.unique(spectrogram.onsets)
    spectrum_list = [spectrogram[spectrogram.onsets == onset] for onset in spectrum_onsets]

    # Reverse onsets and spectrums for right to left overlapping
    if forward == False:
        spectrum_list = spectrum_list[::-1]

    # List of candidate spectra
    spectrum_cand_list = [reduce(lambda spectrum_a, spectrum_b: spectrum_a + spectrum_b,
                                 spectrum_list[i + 1: i + 1 + windows]) for i in range(len(spectrum_list) - windows)]

    # Iterate over the list of spectra and candidate spectra
    for i, (spectrum, spectrum_cand) in enumerate(zip(spectrum_list, spectrum_cand_list)):

        candidates_list = []
        overlaps = []
        matches = []

        # Iterate over the resonances of the current spectrum
        for res in spectrum.elements:

            candidates = spectrum_cand.filter(lambda onset_other, res_other:
                                              res_other.frequency / res.frequency > (1 / freq_ratio) and
                                              res_other.frequency / res.frequency < (freq_ratio))

            candidates_list.append(candidates)

            # Calculate the distances
            if len(candidates) == 0:
                overlaps.append([0])
            elif distance == "harmonic_mean":
                overlaps.append(harmonic_mean_distance(res, candidates, forward))
            elif distance == "residue":
                overlaps.append(residue_distance(res, candidates, forward))
            elif distance == "residue_power":
                overlaps.append(residue_power_distance(res, candidates, forward))
            elif distance == "spectral":
                overlaps.append(spectral_distance(res, candidates))
            elif distance == "residue_transference":
                overlaps.append(residue_distance_transference(res, candidates, spectrum))
            elif distance == "residue_power_transference":
                overlaps.append(residue_power_distance_transference(res, candidates, spectrum))
            else:
                print("Introduce valid distance")
                return

        # Find index of the max overlap   		
        index = maxIndex(overlaps)

        # Iterate while the max overlap is bigger than the min_overlap
        while overlaps[index[0]][index[1]] > (min_overlap if overlap_type == "constant" else
        min_overlap * np.power(10, - (np.abs(spectrum.elements[index[0]].frequency) / 3000))):

            # If resonances can only be matched one to one
            if multiple_match == False:

                # Check that the right resonance is free
                if list(filter(lambda x: x[1] == candidates_list[index[0]][index[1]].elements, matches)) == []:

                    print(spectrum[index[0]].elements)
                    print(candidates_list[index[0]][index[1]].elements)
                    # Confirm match and discard resonance from possible max
                    matches.append((spectrum[index[0]].elements, candidates_list[index[0]][index[1]].elements))
                    overlaps[index[0]][:] = 0

                else:
                    # Discard match and remove max
                    overlaps[index[0]][index[1]] = 0

            # If resonances can be connected from many to one    
            else:

                # Confirm match and discard resonance from possible max
                matches.append((spectrum[index[0]].elements, candidates_list[index[0]][index[1]].elements))
                overlaps[index[0]][:] = 0

            # Find index of the max overlap  
            index = maxIndex(overlaps)

        matches_list.append(matches)

        printProgressBar(i, len(spectrum_list) - 2, prefix='Progress:', suffix='Complete', length=50)

    return matches_list


def overlap_function_intersect(spectrogram, freq_ratio=5, min_overlap=1e-5, overlap_type="linear",
                               distance="residue_power"):
    """Function to match resonances from contiguous spectra (slices) into dynamic resonances based on a resonance distance.
       Forward and backwards matches are obtained allowing multiple match. Intersection of the forward and backwards matches
       is obtained to simplify dynamic resonances.

    Parameters
    ----------
    spectrogram : resonance3.ResonanceSpectrogram
        Spectrogram for which matches between resonances are found
    freq_ratio : float, optional
        Ratio of frequency in times/second relative to the original resonance for which candidates 
        are considered (default is 5)    
    min_overlap : float, optional
        Min overlap between resonances to consider it as a match
    overlap_type : str, optional
        Rate of change of the min_overlap along the frequency. Options are:
            'constant': constant min_overlap 
            'exponential': min_overlap * 10 ^ (Resonance frequency / 3000)
        (default is 'constant')
    distance : str, optional
        Defined distance type to use. Options are:
            harmonic_mean
            residue
            residue_power
            spectral
            residue_transference
            residue_power_transference
        (default is 'residue_power')

    Returns
    -------
    List[List[Tuple]]
        List of matches between resonances as resonance id pairs (Tuple) for each slice (spectrum)
  
    """

    matches_list = []

    # Convert freq ratio from times/second to times/window:
    freq_ratio = 1 + freq_ratio * spectrogram.elements[0].N / spectrogram.sample_rate[0]

    # Obtain list of spectra
    spectrum_onsets = np.unique(spectrogram.onsets)
    spectrum_list = [spectrogram[spectrogram.onsets == onset] for onset in spectrum_onsets]

    for progress, (spectrum_l, spectrum_r) in enumerate(zip(spectrum_list, spectrum_list[1:])):

        candidates_list = [[], []]
        overlaps = [[], []]

        # Candidates for left spectrum are in the right spectrum and viceversa
        for i, (spectrum, spectrum_cand) in enumerate(zip([spectrum_l, spectrum_r], [spectrum_r, spectrum_l])):

            # Iterate over the resonances of the current spectrum
            for res in spectrum.elements:

                # Obtain candidates
                candidates = spectrum_cand.filter(lambda onset_other, res_other:
                                                  res_other.frequency / res.frequency > (1 / freq_ratio) and
                                                  res_other.frequency / res.frequency < (freq_ratio))
                candidates_list[i].append(candidates)

                # Calculate the distances
                if len(candidates) == 0:
                    overlaps[i].append([0])
                elif distance == "harmonic_mean":
                    overlaps[i].append(harmonic_mean_distance(res, candidates, forward=(i == 0)))
                elif distance == "residue":
                    overlaps[i].append(residue_distance(res, candidates, forward=(i == 0)))
                elif distance == "residue_power":
                    overlaps[i].append(residue_power_distance(res, candidates, forward=(i == 0)))
                elif distance == "spectral":
                    overlaps[i].append(spectral_distance(res, candidates))
                elif distance == "residue_transference":
                    overlaps[i].append(residue_distance_transference(res, candidates, spectrum))
                elif distance == "residue_power_transference":
                    overlaps[i].append(residue_power_distance_transference(res, candidates, spectrum))
                else:
                    print("Introduce valid distance")
                    return

        matches = [[], []]

        for i, spectrum in enumerate([spectrum_l, spectrum_r]):

            index = maxIndex(overlaps[i])

            while overlaps[i][index[0]][index[1]] > (min_overlap if overlap_type == "linear" else
            min_overlap * np.power(10, - (np.abs(spectrum.elements[index[0]].frequency) / 3000))):

                # Confirm match and discard resonance from possible max
                # If left to right
                if i == 0:
                    matches[i].append((spectrum[index[0]].elements, candidates_list[i][index[0]][index[1]].elements))
                # If right to left we invert order of keys stored in mathces
                else:
                    matches[i].append((candidates_list[i][index[0]][index[1]].elements, spectrum[index[0]].elements))

                overlaps[i][index[0]][:] = 0

                index = maxIndex(overlaps[i])

        # Intersect matches from left to right and right to left:
        intersected_res = list(set(matches[0]).intersection(set(matches[1])))

        matches_list.append(intersected_res)

        printProgressBar(progress, len(spectrum_list) - 2, prefix='Progress:', suffix='Complete', length=50)

    return matches_list


def horizontalize(index_maps):
    """Function that converts the list of matches between pairs of resonances into a list of dynamic resonances (ids)

    Parameters
    ----------
    index_map : List[List[Tuple]]
        List of matches between resonances as resonance id pairs (Tuple) for each slice (spectrum)
 
    Returns
    -------
    List[List]
        List of dynamic resonances (ids)
  
    """

    dynamic_resonances = []
    previous = {}
    for index_map in index_maps[:]:
        current = {}
        # Extend (or start) any dynamic resonances in previous
        for left_index, right_index in index_map:

            if left_index in previous:
                dyns = previous.pop(left_index)
                # Extend every dynamic resonance
                for dyn in dyns:
                    dyn.append(right_index)

                # If other resonances have already reached the right index resonance
                if right_index in current:
                    current_dyns = current.pop(right_index)
                    current[right_index] = current_dyns + dyns

                # If right index resonance has been only reached from left index resonance
                else:
                    current[right_index] = dyns

            else:
                # If we have a many to one
                if right_index in current:
                    dyns = current.pop(right_index)
                    dyns.append([left_index, right_index])
                    current[right_index] = dyns

                # New dynamic resonance
                else:
                    current[right_index] = [[left_index, right_index]]

        # Any dynamic resonances left in previous are finished
        for dyns in previous.values():
            for dyn in dyns:
                dynamic_resonances.append(dyn)
        previous = current.copy()

    # All dynamic resonances left in current are added
    for dyns in current.values():
        for dyn in dyns:
            dynamic_resonances.append(dyn)

    return dynamic_resonances


def get_dynamic_resonances(spectrogram, min_overlap=5e-1, overlap_type="linear", freq_ratio=5, windows=1,
                           multiple_match=False, mode='intersect', distance="residue_power"):
    """Function to retrieve the dynamic resonances of a spectrogram

    Parameters
    ----------
    spectrogram : resonance3.ResonanceSpectrogram
        Resonance spectrogram from which dynamic resonances are obtained
    min_overlap : float, optional
        Min overlap between resonances to consider it as a match
    overlap_type : str, optional
        Rate of change of the min_overlap along the frequency. Options are:
            'constant': constant min_overlap 
            'exponential': min_overlap * 10 ^ (Resonance frequency / 3000)
        (default is 'constant')
    freq_ratio : float, optional
        Ratio of frequency in times/second relative to the original resonance for which candidates 
        are considered (default is 5)    
    windows : int, optional
        Number of contiguous windows from where candidates are considered (default is 1)
    multiple_match bool, optional
         Allow multiple that multiple resonances can math into a single candidate (default is False)
    mode : str, optional
        Direction in which the matching is carried out. Options are:
            'forward': Forward in time
            'backwards': Backwards in time
            'intersect': Intersection of forward and backwards matching allowing multiple match
        (default is 'constant')
    distance : str, optional
        Defined distance type to use. Options are:
            harmonic_mean
            residue
            residue_power
            spectral
            residue_transference
            residue_power_transference
        (default is 'residue_power')

    Returns
    -------
    resonance3.ResonanceSet
        Resonance set of dynamic resonances. Dynamic resonances are resonance sets of resonances
    """

    # Get index maps of dynamic resonances

    # Left to right matching
    if mode == 'forward':
        index_maps = overlap_function(spectrogram, freq_ratio, min_overlap, overlap_type, windows, multiple_match,
                                      forward=True, distance=distance)

    # Right to left matching
    elif mode == 'backwards':
        index_maps = overlap_function(spectrogram, freq_ratio, min_overlap, overlap_type, windows, multiple_match,
                                      forward=False, distance=distance)
    # Intersection of forward and backard modes
    else:
        index_maps = overlap_function_intersect(spectrogram, freq_ratio, min_overlap, overlap_type, distance=distance)

    # Horizontalize
    dynamic_resonances_idx = horizontalize(index_maps)

    print("after overlap")
    print(dynamic_resonances_idx)
    
    dynamic_spec_onsets = []
    dynamic_resonances = []

    # Convert each dynamic resonance into a ResonanceSet
    # very slow
    for dynamic in dynamic_resonances_idx:
        onsets = spectrogram.onsets[[np.where(spectrogram.elements == res_id)[0][0] for res_id in dynamic]]
        min_onset = np.min(onsets)
        print(min_onset)
        dynamic_spec_onsets.append(min_onset)
        dynamic_resonances.append(ResonanceSet(np.array(dynamic), onsets - min_onset))

    # Convert list of dynamic resonances into a Resonance set
    # HASHING DOESN'T WORK YET
    #return ResonanceSet(np.array([hash(dynamic) for dynamic in dynamic_resonances]), np.array(dynamic_spec_onsets))
    print("i")
    return ResonanceSet(np.array([dynamic for dynamic in dynamic_resonances]), np.array(dynamic_spec_onsets))


def filter_dynamic(dyns, min_length=2, average_distance_thr=0, distance='residue_power', low_cut_freq=None,
                   high_cut_freq=None):
    """Function to filter dynamic resonances. The entire dynamic resonane is removed if any of the filter is True
    Parameters
    ----------
    dyns : resonance3.ResonanceSet
        Resonance set of dynamic resonances.
    min_length : int, optional
        Minimum length of the dynamic resonance (number of resonances)
    average_distance_thr : float, optional
        Minimum average distance between resonances, below this threshold, dynamic resonances are filtered
        (default is 0)
    distance : str, optional
        Defined distance type to use. Options are:
            harmonic_mean
            residue
            residue_power
            spectral
        (default is 'residue_power')
    low_cut_freq : int, optional
        Low cut frequency, high pass frequency filter (default is None)

    high_cut_freq : int, optional
        High cut frequency, low pass frequency filter (default is None)

    Returns
    -------
    resonance3.ResonanceSet
        Filtered resonance set of dynamic resonances. Dynamic resonances are resonance sets of resonances
    """

    # print(len(dyns))

    # Filter dynamic resonances by length (we can also do this with a numpy mask)
    dyns = dyns.filter(lambda onset, dynamic: len(dynamic) >= min_length)

    # print(len(dyns))

    # Filter dynamic resonances by avergae distance
    if average_distance_thr > 0:

        if distance == "harmonic_mean":
            dyns = dyns.filter(lambda onset, dynamic: np.average([harmonic_mean_distance(res_l, res_r, forward=True)
                                                                  for res_l, res_r in
                                                                  zip(dynamic.elements, dynamic.elements[1:])])
                                                      > average_distance_thr)
        elif distance == "residue":
            dyns = dyns.filter(lambda onset, dynamic: np.average([residue_distance(res_l, res_r, forward=True)
                                                                  for res_l, res_r in
                                                                  zip(dynamic.elements, dynamic.elements[1:])])
                                                      > average_distance_thr)
        elif distance == "residue_power":
            dyns = dyns.filter(lambda onset, dynamic: np.average([residue_power_distance(res_l, res_r, forward=True)
                                                                  for res_l, res_r in
                                                                  zip(dynamic.elements, dynamic.elements[1:])])
                                                      > average_distance_thr)
        elif distance == "spectral":
            dyns = dyns.filter(lambda onset, dynamic: np.average([spectral_distance(res_l, res_r)
                                                                  for res_l, res_r in
                                                                  zip(dynamic.elements, dynamic.elements[1:])])
                                                      > average_distance_thr)
        else:
            print("Introduce valid distance")
            return

        # print(len(dyns))

    # High pass filter
    if low_cut_freq != None:
        dyns = dyns.filter(lambda onset, dynamic: np.all(np.array(np.abs(dynamic.frequency)) > low_cut_freq))
        # print(len(dyns))

    # Low pass filter
    if high_cut_freq != None:
        dyns = dyns.filter(lambda onset, dynamic: np.all(np.array(np.abs(dynamic.frequency)) < high_cut_freq))
        # print(len(dyns))

    return dyns


def extend_dynamic_by_amp(dynamic, dyn_onset, spectrum_list, step_size, matched_res, ratio=1.3):
    """Function to extend a dynamic resonance based on amplitude. The dynamic resonance is extended if there exists an additional
    resonance that reduces the error in amplitude at the edge of the window of the reconstructed signal.

    Parameters
    ----------
    dynamic : resonance3.ResonanceSet
        Dynamic resonance to extend.
    dyn_onset : int
        Onset of the dynamic resonance in the entire spectrogram.
    spectrum_list : list
        List of spectra from the original spectrogram
    step_size : int
        Step size of the original dynamic resonance
    matched_res : List
        List of hashes of resonance that already belong to a dynamic resonance
    ratio : float, optional
        Ratio relative to frequency of the dynamic resonance to consider candidates (default is 1.3)

    Returns
    -------
    resonance3.ResonanceSet
        Dynamic resonance (extended if possible)

    bool 
        True if the dynamic resonance has been extended, False otherwise
    """

    dynamic_onsets = np.sort(np.unique(dynamic.onsets))
    dynamic_spectrum_list = [dynamic[dynamic.onsets == onset] for onset in dynamic_onsets]

    for idx, dynamic_spectrum in enumerate(dynamic_spectrum_list):

        # Onset of the slice of inside the dynamic resonance
        onset = dynamic_spectrum.onsets[0]

        if onset < np.max(dynamic_onsets):

            # Reconstruct left resonance
            res_recon = np.zeros(dynamic_spectrum.elements[0].N)
            for res in dynamic_spectrum.elements:
                res_recon += res.reconstruction.real

            # Right resonances
            next_res = dynamic.filter(lambda right_onset, right_res: right_onset == dynamic_onsets[idx + 1])

            # Reconstruct right resonances
            next_res_recon = np.zeros(next_res.elements[0].N)
            if next_res.elements.size == 0:
                return dynamic, False
            for res in next_res.elements:
                next_res_recon += res.reconstruction.real

                # Second right resonances
            if idx + 2 < len(dynamic_onsets):
                next_next_res = dynamic.filter(lambda right_onset, right_res: right_onset == dynamic_onsets[idx + 2])

                # Reconstruction of second right resonance:
                next_next_res_recon = np.zeros(next_next_res.elements[0].N)
                for res in next_next_res.elements:
                    next_next_res_recon += res.reconstruction.real

                    # Candidates spectrum
            spectrum = spectrum_list[1 + (onset + dyn_onset) // step_size]
            candidates = spectrum.filter(lambda cand_onset, cand_res:
                                         cand_res.frequency / np.average(dynamic_spectrum.frequency) > 1 / ratio and
                                         cand_res.frequency / np.average(dynamic_spectrum.frequency) < ratio)  # or
            # np.abs(cand_res.frequency) < 1)

            # Remove candidates that alreeady belong to a dyncamic resonance:
            candidates = candidates.filter(lambda cand_onset, cand_res:
                                           np.all(hash(cand_res) != next_res.elements) and
                                           hash(cand_res) not in matched_res)

            for candidate in candidates.elements:

                current_left_error = np.abs(res_recon[-1].real - next_res_recon[0].real)
                new_left_error = np.abs(res_recon[-1].real - candidate.reconstruction[0].real - next_res_recon[0].real)

                # If left error decreases with correcting resonance
                if current_left_error > new_left_error:

                    # If there is not second right resonance
                    if idx + 2 >= len(dynamic_onsets):
                        dynamic = dynamic + ResonanceSet(np.array([hash(candidate)]), np.array([onset + step_size]))
                        return dynamic, True

                    else:
                        current_right_error = np.abs(next_res_recon[-1].real - next_next_res_recon[0].real)
                        new_right_error = np.abs(
                            next_res_recon[-1].real + candidate.reconstruction[-1].real - next_next_res_recon[0].real)

                        # If right error is decreased with correcting resonance
                        if current_right_error > new_right_error:
                            dynamic = dynamic + ResonanceSet(np.array([hash(candidate)]), np.array([onset + step_size]))
                            return dynamic, True

                        # Check if error correction on the left of the next resonance plus correcting resonance is  
                        # bigger than the error increase on the right of the next resonance plus correcting resonance
                        elif current_left_error - new_left_error > (new_right_error - current_right_error) / 1.5:

                            dynamic = dynamic + ResonanceSet(np.array([hash(candidate)]), np.array([onset + step_size]))
                            return dynamic, True

    return dynamic, False


def extend_dynamic_by_phase(dynamic, dyn_onset, spectrum_list, step_size, matched_res, ratio=1.3):
    """Function to extend a dynamic resonance based on phase. The dynamic resonance is extended if there exists an additional
    resonance that reduces the error in the d coefficient at the edge of the window.

    Parameters
    ----------
    dynamic : resonance3.ResonanceSet
        Dynamic resonance to extend.
    dyn_onset : int
        Onset of the dynamic resonance in the entire spectrogram.
    spectrum_list : list
        List of spectra from the original spectrogram
    step_size : int
        Step size of the original dynamic resonance
    matched_res : List
        List of hashes of resonance that already belong to a dynamic resonance
    ratio : float, optional
        Ratio relative to frequency of the dynamic resonance to consider candidates (default is 1.3)

    Returns
    -------
    resonance3.ResonanceSet
        Dynamic resonance (extended if possible)

    bool 
        True if the dynamic resonance has been extended, False otherwise
    """

    dynamic_onsets = np.sort(np.unique(dynamic.onsets))
    dynamic_spectrum_list = [dynamic[dynamic.onsets == onset] for onset in dynamic_onsets]

    for idx, dynamic_spectrum in enumerate(dynamic_spectrum_list):

        # Onset of the slice inside the dynamic resonance
        onset = dynamic_spectrum.onsets[0]

        # if 1 + (onset + dyn_onset)//step_size < len(spectrum_list):
        if onset < np.max(dynamic.onsets):

            # Right resonance
            next_res = dynamic.filter(lambda right_onset, right_res: right_onset == dynamic_onsets[idx + 1])

            # Second right resonance
            if idx + 2 < len(dynamic_onsets):
                next_next_res = dynamic.filter(lambda right_onset, right_res: right_onset == dynamic_onsets[idx + 2])

            # Spectrum of candidates:
            spectrum = spectrum_list[1 + (onset + dyn_onset) // step_size]

            candidates = spectrum.filter(lambda cand_onset, cand_res:
                                         cand_res.frequency / np.average(dynamic_spectrum.frequency) > 1 / ratio and
                                         cand_res.frequency / np.average(dynamic_spectrum.frequency) < ratio)  # or
            # np.abs(cand_res.frequency) < 1)

            # Remove candidates that alreeady belong to a dyncamic resonance:
            candidates = candidates.filter(lambda cand_onset, cand_res:
                                           np.all(hash(cand_res) != next_res.elements) and
                                           hash(cand_res) not in matched_res)

            shifted_d = np.sum(np.array(dynamic_spectrum.d) * np.exp(-1j * np.array(dynamic_spectrum.w)
                                                                     * np.array(dynamic_spectrum.N) / np.array(
                dynamic_spectrum.sample_rate)))

            if candidates.elements.size > 0:

                current_error = np.abs(shifted_d - np.sum(next_res.d))
                new_error = np.abs(shifted_d - (np.sum(next_res.d) + candidates.d))

                # If second right resonances exist
                if idx + 2 < len(dynamic_onsets):
                    current_error += np.abs(np.sum(next_res.d * np.exp(-1j * np.array(next_res.w)
                                                                       * np.array(next_res.N) / np.array(
                        next_res.sample_rate)))
                                            - np.sum(next_next_res.d))

                    new_error += np.abs(np.sum(next_res.d * np.exp(-1j * np.array(next_res.w)
                                                                   * np.array(next_res.N) / np.array(
                        next_res.sample_rate)))
                                        + candidates.d * np.exp(-1j * np.array(candidates.w)
                                                                * np.array(candidates.N) / np.array(
                        candidates.sample_rate))
                                        - np.sum(next_next_res.d))

                # Obtain the minimum error
                new_error_idx = np.argmax(- new_error)

                # if error[error_idx] < np.abs(shifted_d - np.sum(next_res.d)):
                if new_error[new_error_idx] < current_error:
                    dynamic = dynamic + ResonanceSet(np.array([hash(candidates.elements[new_error_idx])]),
                                                     np.array([onset + step_size]))
                    return dynamic, True

    return dynamic, False


def extend_dynamic_by_decay(dynamic, dyn_onset, spectrum_list, step_size, matched_res, f0):
    """Function to extend a dynamic resonance based on decay. The dynamic resonance is extended with all the resonances in the candidates
    range that point inwards (damping above center resonance, ramping below center resonance) if the dynamic resonance is 
    decreasing in frequency and vice versa. 

    # TO DO: adapt method to accept f0 as a list

    Parameters
    ----------
    dynamic : resonance3.ResonanceSet
        Dynamic resonance to extend.
    dyn_onset : int
        Onset of the dynamic resonance in the entire spectrogram.
    spectrum_list : list
        List of spectra from the original spectrogram
    step_size : int
        Step size of the original dynamic resonance
    matched_res : List
        List of hashes of resonance that already belong to a dynamic resonance
    f0 : float, optional
        Fundamental frequency of the signal

    Returns
    -------
    resonance3.ResonanceSet
        Dynamic resonance (extended if possible)

    bool 
        False since this extension method only requires to extend once each dynamic resonance
    """

    new_dynamic = dynamic

    for previous, current, current_onset in zip(dynamic.elements, dynamic.elements[1:], dynamic.onsets[1:]):

        # Spectrum of candidates:
        spectrum = spectrum_list[(current_onset + dyn_onset) // step_size]

        # Obtain candidates based on ratio
        # candidates = spectrum.filter(lambda cand_onset, cand_res: 
        #                             cand_res.frequency / current.frequency > 1 / ratio and 
        #                             cand_res.frequency / current.frequency < ratio)

        # Obtain candidates based on f0 frequency window
        candidates = spectrum.filter(lambda cand_onset, cand_res:
                                     cand_res.frequency < current.frequency + f0 and
                                     cand_res.frequency > current.frequency - f0)

        # Remove candidates that alreeady belong to a dyncamic resonance:
        candidates = candidates.filter(lambda cand_onset, cand_res: hash(cand_res) not in matched_res)

        # If dynamic resonance is decreassing in freq
        if previous.frequency > current.frequency:
            higher = candidates.filter(lambda cand_onset, cand_res: np.array(cand_res.w).imag < 0
                                                                    and cand_res.frequency > current.frequency)
            if len(higher) > 0:
                # Add higher frequency resonances
                new_dynamic += ResonanceSet(higher.elements, np.array([current_onset] * len(higher)))

            # Add lower frequency resonances
            lower = candidates.filter(lambda cand_onset, cand_res: np.array(cand_res.w).imag > 0
                                                                   and cand_res.frequency < current.frequency)
            if len(lower) > 0:
                new_dynamic += ResonanceSet(lower.elements, np.array([current_onset] * len(lower)))

        # If dynamic resonance is increasing in freq   
        else:
            # Add higher frequency resonances
            higher = candidates.filter(lambda cand_onset, cand_res: np.array(cand_res.w).imag > 0
                                                                    and cand_res.frequency > current.frequency)
            if len(higher) > 0:
                new_dynamic += ResonanceSet(higher.elements, np.array([current_onset] * len(higher)))

            # Add lower frequency resonances
            lower = candidates.filter(lambda cand_onset, cand_res: np.array(cand_res.w).imag < 0
                                                                   and cand_res.frequency < current.frequency)
            if len(lower) > 0:
                new_dynamic += ResonanceSet(lower.elements, np.array([current_onset] * len(lower)))

    return new_dynamic, False


def extend_dynamic_res(spectrogram, dyns, step_size, extension_type='amplitude', ratio=1.3, f0=None):
    """Function to extend a set of dynamic resonances based on one of the three defined extension methods.

    Parameters
    ----------
    spectrogram : resonance3.ResonanceSpectrogram
        ORiginal resonance spectrogram
    dyns : resonance3.ResonanceSet
        Resonance set of dynamic resonances to extend.
    step_size : int
        Step size of the original dynamic resonance
    matched_res : List
        List of hashes of resonance that already belong to a dynamic resonance
    extension_type : str, optional
        Extension method used to extend the dynamic resonances. Options are:
            'amplitude': Extend by amplitude
            'phase': Extend by phase
            'decay': Extend by decay
        (default is 'amplitude')
    ratio : float, optional
        Ratio relative to frequency of the dynamic resonance to consider candidates (default is 1.3)
    f0 : float, optional
        Fundamental frequency of the signal. Only required for extension method 'decay' (default is None)

    Returns
    -------
    resonance3.ResonanceSet
        Set of extended dynamic resonances
    """

    spectrum_list = [spectrogram[spectrogram.onsets == onset] for onset in np.unique(spectrogram.onsets)]

    for dynamic, dyn_onset in zip(dyns.elements, dyns.onsets):

        # print("Extending")
        modified = True

        while modified:

            dynamic_old = dynamic
            matched_res = np.hstack([dyn.elements for dyn in dyns.elements])
            # Extend by decreasing error in amplitude in time at the edge of the window
            if extension_type == 'amplitude':
                dynamic, modified = extend_dynamic_by_amp(dynamic, dyn_onset, spectrum_list, step_size, matched_res,
                                                          ratio)
            # Extend by decreasing error in d coefficient at the edge of the window
            elif extension_type == 'phase':
                dynamic, modified = extend_dynamic_by_phase(dynamic, dyn_onset, spectrum_list, step_size, matched_res,
                                                            ratio)
            # Extend by decay depending on increasing or decreasing frequency of the dynamic resonance
            elif extension_type == 'decay':
                if f0 == None:
                    print("Introduce a valid f0 for decay extension")
                else:
                    dynamic, modified = extend_dynamic_by_decay(dynamic, dyn_onset, spectrum_list, step_size,
                                                                matched_res, f0)
            else:
                print("Introduce a valid extension method")

            dyns = ResonanceSet(np.append(dyns.elements[dyns.elements != hash(dynamic_old)], hash(dynamic)),
                                np.append(dyns.onsets[dyns.elements != hash(dynamic_old)], dyn_onset))

    return dyns


def plot_dynamic(spectrogram, dyns, signal=None, min_freq=0, max_freq=3000, hide_dynamic=False,
                 plot_signal=False, plot_fourier=False, scale=1, size='distance', distance='residue_power'):
    """Function to plot dynamic resonances spectrogram over the orginal discrete spectrogram
    
    Parameters
    ----------
    spectrogram : resonance3.ResonanceSpectrogram
        Resonance spectrogram to plot
    dyns : resonance3.ResonanceSet
        Resonance set of dynamic resonances.
    signal : numpy.ndarray, optional
        Time signal, used when the Fourier Transform is added to the plot (default is None)
    min_freq : int, optional
        Lower bound frequency (Hz) of the plot (default is 0)
    max_freq : int, optional
        Upper bound frequency (Hz) of the plot (default is 3000)
    hide_dynamic : bool, optional
        If True, dynamic resonances are not plotted (default is False)
    plot_signal
        If True, the time signal is added to the plot (default is False)
    plot_fourier bool, optional
         If True, the Fast Fourier Spectrogram is added to the plot (default is False)
    scale float, optional
         Scale size of resonances (points of the scatterplot) (default is 1)
    size : str, optional
        Size of resonances from the dynamic resonances (points of the scatterplot). Options are:
            'distance': The size of the points is inversely proportional to the distance between the resonances of each dynamic resonance
            'power': The size of the points is proportional to the power of the resonances.
        (default is 'distance')	
    distance : str, optional
        If size is 'distance', distance specifies the defined distance to calculate between resonances of each dynamic resonance. Options are:
            harmonic_mean
            residue
            residue_power
            spectral
        (default is 'residue_power')	

    """

    fig = plt.figure(figsize=(15, 8))

    ax = fig.add_subplot(111, label="1")

    # Original spectrogram    

    # if not hide_dynamic:
    #    ax.scatter(spectrogram.onsets / spectrogram.sample_rate, spectrogram.frequency, s = 0.2 * scale, c = 'black', zorder = 2)
    # else:
    ax.scatter(spectrogram.onsets / spectrogram.sample_rate, spectrogram.frequency,
               s=np.array(spectrogram.power) * 10 * scale, c='black', zorder=2)

    # Dynamic resonances spectrogram
    if not hide_dynamic:
        
        for dynamic, onset in zip(dyns.elements, dyns.onsets): #REPLACED dyns.elements by dyns
            
            print("B")
            print(dynamic)
            print("--------------------")
            print(dynamic.frequency[0])
            print("--------------------")
            print(spectrogram)
            
            if dynamic.frequency[0] > min_freq and dynamic.frequency[0] < 2 * max_freq:
                # onset_s = np.array(dynamic.onsets)/spectrogram.sample_rate[0] -> Onset of resonances in dynamic as N0 ins
                onset_s = (onset + dynamic.onsets) / spectrogram.sample_rate[0]

                if size == 'distance':
                    if distance == "harmonic_mean":
                        s = np.array([harmonic_mean_distance(res_l, res_r, forward=True) / 1e6
                                      for res_l, res_r in zip(dynamic.elements, dynamic.elements[1:])])
                    elif distance == "residue":
                        s = np.array([residue_distance(res_l, res_r, forward=True)
                                      for res_l, res_r in zip(dynamic.elements, dynamic.elements[1:])])
                    elif distance == "residue_power":
                        s = np.sqrt(np.array([residue_power_distance(res_l, res_r, forward=True)
                                              for res_l, res_r in zip(dynamic.elements, dynamic.elements[1:])])) / 100
                    elif distance == "spectral":
                        s = np.array([spectral_distance(res_l, res_r)
                                      for res_l, res_r in zip(dynamic.elements, dynamic.elements[1:])]) / 100
                    ax.scatter(onset_s[:-1], dynamic.frequency[:-1], s=np.sqrt(s * 1e8) * scale, alpha=0.5, zorder=4)

                else:
                    ax.scatter(onset_s, dynamic.frequency, s=np.array(dynamic.power) * 10 * scale, alpha=1, zorder=4)

                # Add connections between points with plot:
                ordering_idx = np.argsort(onset_s)
                onset_s = onset_s[ordering_idx]
                frequency = np.array(dynamic.frequency)[ordering_idx]

                ax.plot(onset_s, frequency, zorder=3)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_ylim((min_freq, max_freq))

    # Add Fourier representation:
    if plot_fourier:
        if signal is not None:
            f, t, Sxx = spsignal.spectrogram(signal, spectrogram.sample_rate[0])
            ax.pcolormesh(t, f, np.log(Sxx), shading='gouraud', zorder=1)
            # ax.pcolormesh(ts, ws / (2 * np.pi), 5000 * np.square(np.abs(ds)), shading='gouraud', zorder = 1)
            # ax.contourf(ts, ws / (2 * np.pi), np.log(ds), cmap='viridis', levels=100, zorder = 1)
        else:
            print("You must pass the original signal to obtain Fourier representation")

    # Add time signal:
    if plot_signal and signal is not None:
        ax2 = fig.add_subplot(111, label="2", frame_on=False)
        ax2.set_axisbelow(True)

        time = np.arange(0, len(signal) / spectrogram.sample_rate[0], 1 / spectrogram.sample_rate[0])

        # Adapt color based on fourier plot
        if plot_fourier:
            ax2.plot(time, signal, alpha=0.5, linewidth=0.6, c='#800080')
        else:
            ax2.plot(time, signal, alpha=0.5, linewidth=0.6)

        ax2.set_xticks([])
        ax2.set_ylabel('Amplitude')
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position('right')
    plt.title('Dynamic Resonance Spectrogram')


def plot_dynamic_overlapped(spectrogram_A, dyns_A, spectrogram_B, dyns_B, spectrogram, signal, dyns,
                            min_freq=0, max_freq=3000, plot_fourier=False, scale=1):
    """Function to plot 3 dynamic resonances spectrogram. This function is developped for representation purposes
    for source separation. It plots the dynamic resonance specrtogram of the mixture of signals and the two dynamic
    resonance spectrograms of the orignal compnents.
    
    Parameters
    ----------
    spectrogram_A : resonance3.ResonanceSpectrogram
        Resonance spectrogram of first the component
    dyns_A : resonance3.ResonanceSet
        Resonance set of dynamic resonances of the first component.
    spectrogram_B : resonance3.ResonanceSpectrogram
        Resonance spectrogram of the second component
    dyns_B : resonance3.ResonanceSet
        Resonance set of dynamic resonances of the second component.
    spectrogram : resonance3.ResonanceSpectrogram
        Resonance spectrogram of the mixture
    signal : numpy.ndarray
        Time signal of the mixture
    dyns : resonance3.ResonanceSet
        Resonance set of dynamic resonances of the mixture.
    min_freq : int, optional
        Lower bound frequency (Hz) of the plot (default is 0)
    max_freq : int, optional
        Upper bound frequency (Hz) of the plot (default is 3000)
    plot_fourier bool, optional
         If True, the Fast Fourier Spectrogram of the mixture is added to the plot (default is False)
    scale float, optional
         Scale size of resonances (points of the scatterplot) (default is 1)
    """

    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111, label="1")

    # Original spectrogram
    # ax.scatter(spectrogram.onsets / spectrogram.sample_rate, spectrogram.frequency, s = np.array(spectrogram.power) * 10 * scale, c = 'black', zorder = 2)

    colors = ['lime', 'white', 'red']
    transparency = [1, 1, 0.5]
    signal_name = ['Signal A', 'Signal B', 'Mixture']

    # Dynamic resonances spectrogram
    for idx, (spectrogram_, dyns_) in enumerate(
            zip([spectrogram_A, spectrogram_B, spectrogram], [dyns_A, dyns_B, dyns])):

        label = True

        for dynamic, onset in zip(dyns_.elements, dyns_.onsets):

            if dynamic.frequency[0] > min_freq and dynamic.frequency[0] < 2 * max_freq:
                onset_s = (onset + dynamic.onsets) / spectrogram_.sample_rate[0]
                ax.scatter(onset_s, dynamic.frequency, s=np.array(dynamic.power) * 12 * scale,
                           alpha=transparency[idx], zorder=idx + 2, c=colors[idx])

                # Add connections between points with plot:
                ordering_idx = np.argsort(onset_s)
                onset_s = onset_s[ordering_idx]
                frequency = np.array(dynamic.frequency)[ordering_idx]

                ax.plot(onset_s, frequency, zorder=idx + 2, c=colors[idx], alpha=transparency[idx],
                        label=signal_name[idx] if label else "")

                label = False

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_ylim((min_freq, max_freq))
    ax.set_facecolor('navy')

    # Add Fourier representation:
    if plot_fourier:
        if signal is not None:
            f, t, Sxx = spsignal.spectrogram(signal, spectrogram.sample_rate[0], window='hamming')
            ax.pcolormesh(t, f, np.log(Sxx), shading='gouraud', zorder=1, cmap='viridis',
                          vmax=1.3 * (np.max(np.log(Sxx))))
        else:
            print("You must pass the original signal to obtain Fourier representation")

    plt.legend()
    plt.title('Dynamic Resonance Spectrogram')


""" Density Spectrogrma related funtions """


def continuous_density_spectrogram(spectrogram, min_freq=0, max_freq=0, slices_overlap=7, freq_overlap=200,
                                   precision=500):
    """Function to obtain the continuous density spectrogram

    Parameters
    ----------
    spectrogram : resonance3.ResonanceSpectrogram
        Resonance spectrogram from which the density spectrogram is obtained
    min_freq : int, optional
        Lower bound frequency (Hz) of the density spectrogram (default is 0)
    max_freq : int, optional
        Upper bound frequency (Hz) of the density spectrogram (default is 0). If max_freq == 0, the upper bound is the sample rate / 2
    slices_overlap : int, optional
        Number of slices that contribute to the density of one point (default is 7)
    freq_overlap : int, optional
        Frequency range of considered resonances that contribute to the density of one point (default is 200)
    precision : int, optional
        Number of equally spaced (in frequency) points for which the density is calculated (default is 500)

    Returns
    -------
    numpy.ndarray
        Continuous density spectrogram
    """

    sample_rate = spectrogram.sample_rate[0]
    slice_sigma = 1 + slices_overlap / 10
    time_window = spsignal.windows.gaussian(slices_overlap, slice_sigma, sym=True)
    spectrum_onsets = np.unique(spectrogram.onsets)
    spectrum_list = [spectrogram[spectrogram.onsets == onset] for onset in spectrum_onsets]

    if min_freq == 0 and max_freq == 0:
        domain = np.arange(0, np.pi * sample_rate, np.pi * sample_rate / precision)
    else:
        domain = np.arange(2 * np.pi * min_freq, 2 * np.pi * max_freq, 2 * np.pi * (max_freq - min_freq) / precision)

    density_spectrogram = []

    # i indexes the slice
    for i in range(len(spectrum_list)):
        density = []
        # j indexes the Gaussian window
        # i + j - slice_overlaps//2 indexes the corresponding slice of the spectrum for the j Gaussian slice
        for j in range(slices_overlap):

            if i + j - slices_overlap // 2 >= 0 and i + j - slices_overlap // 2 < len(spectrum_list):
                # Calculate the density
                density.append(
                    time_window[j] * np.array([np.sum(np.square(np.abs(spectrum_list[i + j - slices_overlap // 2].d)) *
                                                      # np.array(spectrum_list[i + j - slices_overlap//2].power) * # Power weighted
                                                      voigt(ws_eval - np.array(
                                                          spectrum_list[i + j - slices_overlap // 2].w).real,
                                                            freq_overlap * 2 * np.pi, np.abs(np.array(
                                                              spectrum_list[i + j - slices_overlap // 2].w).imag))  # /
                                                      # np.abs(np.array(spectrum_list[i + j - slices_overlap//2].w).imag) # Divide by w.imag
                                                      )
                                               for ws_eval in domain]))

        density_spectrogram.append(np.sum(np.array(density), axis=0))

    return np.array(density_spectrogram)


def discrete_density_spectrogram(spectrogram, slices_overlap=7, freq_overlap=200):
    """Function to obtain the discrete density spectrogram

    Parameters
    ----------
    spectrogram : resonance3.ResonanceSpectrogram
        Resonance spectrogram from which the density spectrogram is obtained
    slices_overlap : int, optional
        Number of slices that contribute to the density of one point (default is 7)
    freq_overlap : int, optional
        Frequency range of considered resonances that contribute to the density of one point (default is 200)

    Returns
    -------
    numpy.ndarray
        Discrete density spectrogram
    """

    sample_rate = spectrogram.sample_rate[0]
    slice_sigma = 1 + slices_overlap / 10
    time_window = spsignal.windows.gaussian(slices_overlap, slice_sigma, sym=True)
    spectrum_onsets = np.unique(spectrogram.onsets)
    spectrum_list = [spectrogram[spectrogram.onsets == onset] for onset in spectrum_onsets]

    density_spectrogram = []

    # i indexes the slice
    for i in range(len(spectrum_list)):
        density = []
        # j indexes the Gaussian window
        # i + j - slice_overlaps//2 indexes the corresponding slice of the spectrum for the j Gaussian slice
        for j in range(slices_overlap):

            if i + j - slices_overlap // 2 >= 0 and i + j - slices_overlap // 2 < len(spectrum_list):
                # Calculate the density
                density.append(
                    time_window[j] * np.array([np.sum(np.square(np.abs(spectrum_list[i + j - slices_overlap // 2].d)) *
                                                      # np.array(spectrum_list[i + j - slices_overlap//2].power) * # Power weighted
                                                      voigt(ws_eval - np.array(
                                                          spectrum_list[i + j - slices_overlap // 2].w).real,
                                                            freq_overlap * 2 * np.pi, np.abs(np.array(
                                                              spectrum_list[i + j - slices_overlap // 2].w).imag))  # /
                                                      # np.abs(np.array(spectrum_list[i + j - slices_overlap//2].w).imag) # Divide by w.imag
                                                      )
                                               for ws_eval in np.array(spectrum_list[i].w).real]))

        density_spectrogram.append(np.sum(np.array(density), axis=0))

    return density_spectrogram


def plot_continuous_density_spectrogram(density_spectrogram, spectrogram, precision=500, min_freq=0, max_freq=None,
                                        threshold=0):
    """Function to plot the continuous density spectrogram

    Parameters
    ----------
    density_spectrogram : numpy.ndarray
        Continuous density spectrogram to plot
    spectrogram : resonance3.ResonanceSpectrogram
        Original resonance spectrogram
    precision : int, optional
        Number of equally spaced (in frequency) points for which the density has been calculated (default is 500)    
    min_freq : int, optional
        Lower bound frequency (Hz) of the density spectrogram (default is 0)
    max_freq : int, optional
        Upper bound frequency (Hz) of the density spectrogram (default is None).
    threshold float, optional
         Minimum density of the point to be included in the plot (default is 0)
    """

    fig, ax = plt.subplots(figsize=(15, 8))

    sample_rate = spectrogram.sample_rate[0]

    t = np.unique(spectrogram.onsets) / sample_rate

    if max_freq == None:
        domain = np.arange(0, np.pi * sample_rate, np.pi * sample_rate / precision)
    else:
        domain = np.arange(2 * np.pi * min_freq, 2 * np.pi * max_freq, 2 * np.pi * (max_freq - min_freq) / precision)

    masked_density = density_spectrogram.T

    if threshold > 0:
        masked_density[masked_density < threshold] = 0

    ax.contourf(t, domain / (2 * np.pi), np.cbrt(masked_density), cmap='viridis', levels=500)

    ax.set_xlim(0, np.max(t))
    if max_freq == None:
        ax.set_ylim(min_freq, sample_rate // 2)
    else:
        ax.set_ylim(min_freq, max_freq)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Continuous Density Resonance Spectrogram')


def plot_discrete_density_spectrogram(density_spectrogram, spectrogram, min_freq=0, max_freq=None, threshold=0,
                                      resize_factor=1):
    """Function to plot the discrete density spectrogram

    Parameters
    ----------
    density_spectrogram : numpy.ndarray
        Discrete density spectrogram to plot
    spectrogram : resonance3.ResonanceSpectrogram
        Original resonance spectrogram
    precision : int, optional
        Number of equally spaced (in frequency) points for which the density has been calculated (default is 500)    
    signal : numpy.ndarray, optional
        Time signal, used when the Fourier Transform is added to the plot (default is None)
    min_freq : int, optional
        Lower bound frequency (Hz) of the density spectrogram (default is 0)
    max_freq : int, optional
        Upper bound frequency (Hz) of the density spectrogram (default is None)
    threshold float, optional
         Minimum density of the point to be included in the plot (default is 0)
    resize_factor float, optional
         Scale size of resonances (points of the scatterplot) (default is 1)
    """

    fig, ax = plt.subplots(figsize=(15, 8))

    sample_rate = spectrogram.sample_rate[0]

    densities = np.hstack(density_spectrogram)

    t = np.hstack(spectrogram.onsets) / sample_rate
    freqs = np.hstack(spectrogram.w).real / (2 * np.pi)

    # Remove resonances with density < threshold
    if threshold > 0:
        densities[densities < threshold] = 0

    ax.scatter(t, freqs, s=resize_factor * 1e3 * np.cbrt(densities), c=resize_factor * 1e3 * np.cbrt(densities))
    ax.set_xlim(0, np.max(t))

    if max_freq == None:
        ax.set_ylim(min_freq, sample_rate // 2)
    else:
        ax.set_ylim(min_freq, max_freq)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Discrete Density Resonance Spectrogram')


def filter_by_density(spectrogram, density_spectrogram, threshold):
    """Function to filter resonances by density

    Parameters
    ----------

    spectrogram : resonance3.ResonanceSpectrogram
        Original resonance spectrogram
    density_spectrogram : numpy.ndarray
        Discrete density spectrogram to plot
    threshold float, optional
         Resonances with a density lower than the threshold are filtered
    
    Returns
    -------
    numpy.ndarray
        Filtered discrete density spectrogram

    """
    mask = np.hstack(density_spectrogram) > threshold
    return ResonanceSet(spectrogram.elements[mask], spectrogram.onsets[mask])


""" Fundamental frequency (f0) estimation realted functions """


def get_f0(spectrogram, ratio=0.1, n_sources=1):
    """Function to obtain the fundamental frequency of n harmonic sources of the original signal.
    Currently only working for 2 sources.

    TO DO: Adapt method to correct more than 2 sources

    Parameters
    ----------

    spectrogram : resonance3.ResonanceSpectrogram
        Resonance spectrogram for which f0 is estimated
    ratio : float, optional
        Ratio realtive to the freuqnecy of a resonance to include other resonances as contributing to the harmonicity.
        These candidates resonances must be multiples or divisors in frequency of the original resonance (d)
    n_sources int, optional
         Number of sound sources in the original signal (default is 2)

    Returns
    -------
    List[List]
        List of n fundamental frequencies (f0) of the signal

    """

    spectrum_onsets = np.unique(spectrogram.onsets)
    spectrum_list = [spectrogram[spectrogram.onsets == onset] for onset in spectrum_onsets]

    f0_s = []

    for spectrum in spectrum_list:

        spectrum_f0 = []
        sources = n_sources

        # Calculate the harmonicity
        harmonicity = np.array([np.sum(spectrum[np.logical_and(np.abs(spectrum.frequency / res.frequency) >= 1 - ratio,
                                                               np.logical_or(
                                                                   np.modf(np.abs(spectrum.frequency / res.frequency))[
                                                                       0] < ratio,
                                                                   np.modf(np.abs(spectrum.frequency / res.frequency))[
                                                                       0] > 1 - ratio))].power)
                                if np.abs(res.frequency) > 50 else 0 for res in spectrum.elements])

        # Obtain the n_sources maximum harmonicities that are at least further than the frequency ratio
        while sources > 0 and not np.all(harmonicity == 0):

            candidate_f0 = np.abs(spectrum.frequency[np.argmax(harmonicity)])

            if not spectrum_f0:
                spectrum_f0.append(candidate_f0)
                sources -= 1

            else:

                existing = []

                for existing_f0 in spectrum_f0:
                    if existing_f0 > candidate_f0:
                        existing.append(np.modf(existing_f0 / candidate_f0)[0] < ratio or
                                        np.modf(existing_f0 / candidate_f0)[0] > 1 - ratio)
                    else:
                        existing.append(np.modf(candidate_f0 / existing_f0)[0] < ratio or
                                        np.modf(candidate_f0 / existing_f0)[0] > 1 - ratio)

                if not np.any(existing):
                    spectrum_f0.append(candidate_f0)
                    sources -= 1

            harmonicity[np.argmax(harmonicity)] = 0

        f0_s.append(spectrum_f0)

    # Match and correct f0s when n_sources > 1
    if n_sources > 1:

        # Match the correct f0 from the two detected sources

        matched_f0_s = match_correct_f0(f0_s)

        # Correct f0 

        corrected_f0_s = []

        for matched_f0 in np.array(matched_f0_s).transpose():
            corrected_f0_s.append(correct_f0(spectrogram, matched_f0))

        return corrected_f0_s

    # Return single f0, when n_sources = 1
    # TO DO apply correction of f0 to a single source
    else:
        return np.array(f0_s).transpose()


def match_correct_f0(f0_s):
    """Function to assign to each source the corresponding f0 out of the n_sources candidates.
    Currently only working for 2 sources.

    TO DO: Adapt method to correct more than 2 sources

    Parameters
    ----------

    f0_s : 
        List of n_sources candidates f0s

    Returns
    -------
    List
        Correctly matched f0s to their corresponding sources

    """

    ordered_f0_s = [f0_s[0]]

    for idx, f0 in enumerate(f0_s[1:]):

        f0s_to_f0s = []

        # Calculate spectral harmonic distance to the previous resonance in the ordered_f0_s list
        for ordered_f0 in ordered_f0_s[idx]:

            distances = []
            for single_f0 in f0:

                if ordered_f0 > single_f0:
                    distance = np.modf(ordered_f0 / single_f0)[0]
                    if distance < 0.5:
                        distances.append(distance)
                    else:
                        distances.append(1 - distance)

                else:
                    distance = np.modf(single_f0 / ordered_f0)[0]
                    if distance < 0.5:
                        distances.append(distance)
                    else:
                        distances.append(1 - distance)

            f0s_to_f0s.append(distances)

        f0s_to_f0s = np.array(f0s_to_f0s)

        min_distance = np.where(- f0s_to_f0s == np.amax(- f0s_to_f0s))

        # Only for two sources
        if min_distance[0][0] == 0:
            if min_distance[1][0] == 0:
                ordered_f0_s.append(f0)
            else:
                ordered_f0_s.append(list(np.flip(f0)))
        else:
            if min_distance[1][0] == 1:
                ordered_f0_s.append(f0)
            else:
                ordered_f0_s.append(list(np.flip(f0)))

    return ordered_f0_s


def correct_f0(spectrogram, f0):
    """Function to correct the f0 from a source, assigning to each f0 the closest candidate to the common divisor.

    Parameters
    ----------

    f0 : 
        List of candidate f0

    Returns
    -------
    List
        Corrected f0

    """

    spectrum_onsets = np.unique(spectrogram.onsets)
    spectrum_list = [spectrogram[spectrogram.onsets == onset] for onset in spectrum_onsets]

    for i in range(len(range(len(f0) - 4))):

        context_f0 = f0[i:i + 3]
        next_f0 = f0[i + 3]
        spectrum = spectrum_list[i + 3]

        # Calculate standard deviation and average of the f0 of the 4 previous slices
        std = np.std(context_f0)
        avg = np.average(context_f0)

        # If the standard deviation is relatively small
        if std / avg < 0.3:

            frequencies = np.array(spectrum.frequency)

            # Find new candidates
            candidates = frequencies[np.logical_and(frequencies > 0.5 * avg, frequencies < 1.5 * avg)]
            candidates = candidates[candidates != 0]

            distances = []

            for candidate in candidates:

                # Calculate distance
                if candidate > next_f0:
                    distance = np.modf(candidate / next_f0)[0]
                else:
                    distance = np.modf(next_f0 / candidate)[0]

                # Correct distance if bigger than 0.5
                if distance < 0.5:
                    distances.append(distance)
                else:
                    distances.append(1 - distance)

            if candidates.size != 0:
                f0[i + 3] = candidates[np.argmax(- np.array(distances))]

    return f0


def plot_f0(spectrogram, f0_s, min_freq=0, max_freq=3000):
    """Function to plot fundamental frequencies of a signal over the discrete spectrogram

    Parameters
    ----------
    spectrogram : resonance3.ResonanceSpectrogram
        Original resonance spectrogram
    f0_s : List[List]
        List of the fundamental frequencies 
    min_freq : int, optional
        Lower bound frequency (Hz) of the density spectrogram (default is 0)
    max_freq : int, optional
        Upper bound frequency (Hz) of the density spectrogram (default is 3000)
    """

    onsets = np.unique(spectrogram.onsets)

    fig = plt.figure(figsize=(15, 8))

    ax = fig.add_subplot(111, label="1")

    # Original spectrogram      
    ax.scatter(spectrogram.onsets / spectrogram.sample_rate, spectrogram.frequency, s=np.array(spectrogram.power) * 10,
               c=np.array(spectrogram.power), zorder=1)

    # spectrum_list = [spectrogram[spectrogram.onsets == onset] for onset in onsets]
    # max_powers_f = [np.abs(spectrum.elements[np.argmax(spectrum.power)].frequency) for spectrum in spectrum_list]

    # plt.scatter(onsets/spectrogram.sample_rate[0], max_powers_f, s = 100, c = 'green', label = 'Fundamental frequency')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_ylim((min_freq, max_freq))

    plt.title('Resonance Spectrogram')

    colors = ['red', 'lime', 'deepskyeblue']

    for f0, color in zip(f0_s, colors):
        plt.scatter(onsets / spectrogram.sample_rate[0], np.abs(f0), c=color, label='Fundamental frequency')
    plt.legend(loc=1)


""" Source separation and noise reduction functions """


def source_separation(spectrogram, f0_s, max_dist=0.05, att_factor=20):
    """Function to separate two harmonic sources from the original signal given the fundamental frequency of each source.
    Currently only working for 2 sources.

    TO DO: Adapt method to separate more than 2 sources. It only requires adapting sources_spectrogram and sources_spectrum
    store n_sources instead of just 2

    Parameters
    ----------

    spectrogram : resonance3.ResonanceSpectrogram
        Resonance spectrogram for which f0 is estimated
    f0_s : List[List]
        List of the fundamental frequencies 
    max_dist : float, optional
        Max distance in frequency of a resonance to the closest integer multiple of the f0 so that no attenuation is applied.
        (default is 0.05)
     att_factor float, optional
        Attenuation factor applied to the distant resonances. The attenuation is proportional to the distance in
        frequency to the closest integer multiple of the f0 (default is 20)

    Returns
    -------
    List[List]
        List of n fundamental frequencies (f0) of the signal

    """

    sources_spectrogram = [[], []]

    spectrum_onsets = np.unique(spectrogram.onsets)
    spectrum_list = [spectrogram[spectrogram.onsets == onset] for onset in spectrum_onsets]

    for idx_0, (f0_pair, spectrum, onset) in enumerate(zip(np.array(f0_s).transpose(), spectrum_list, spectrum_onsets)):

        sources_spectrum = [[], []]

        for idx_1, f0 in enumerate(f0_pair):

            # Discard fundamental if the corresponding resonance has a small power
            if np.array(spectrum.power)[np.array(spectrum.frequency) == f0] > 1e-5:

                for res in spectrum.elements:

                    # Calculate distance to the closest integer muntiple of the f0
                    distance = 0

                    if res.frequency > f0:
                        distance = np.abs(np.modf(res.frequency / f0)[0])

                    else:
                        distance = np.abs(np.modf(f0 / res.frequency)[0])

                    if distance > 0.5:
                        distance = 1 - distance

                    # Add res to the corresponding source, if distance < max_distance, res is added 
                    # without attenuation, otherwise, res is attenuated
                    sources_spectrum[idx_1].append(Resonance(res.d if distance < max_dist
                                                             else res.d * (np.exp(- distance * att_factor)),
                                                             res.w, res.z, res.N, res.sample_rate))

                sources_spectrogram[idx_1].append(ResonanceSet(np.array([hash(res) for res in sources_spectrum[idx_1]]),
                                                               np.array([onset] * len(sources_spectrum[idx_1]))))

    return np.sum(sources_spectrogram[0]), np.sum(sources_spectrogram[1])


def denoise_by_power(spectrogram, power_factor=0.5):
    """Function to filter noisy resonance by their lower power in relation to the average power of all resonances.

    Parameters
    ----------

    spectrogram : resonance3.ResonanceSpectrogram
        Resonance spectrogram to denoise
    power_factor : float, optional
        Factor that multiplies the average power to estimate the minimum power thresohld. (default is 0.5)

    Returns
    -------
    resonance3.ResonanceSpectrogram
        Resonance spectrogram denoised by power

    """
    print(spectrogram)

    average_power = np.average(spectrogram.power) * power_factor
    return spectrogram.filter(lambda onset, res: res.power > average_power)


def denoise_by_density(spectrogram, slices_overlap=7, freq_overlap=250, density_threshold=None, density_factor=5):
    """Function to denoise a spectrogram based on the spectral density. Resonances with a low density are attenuated.

    Parameters
    ----------

    spectrogram : resonance3.ResonanceSpectrogram
        Resonance spectrogram to denoise
    slices_overlap : int, optional
        Number of slices that contribute to the density of one point (default is 7)
    freq_overlap : int, optional
        Frequency range of considered resonances that contribute to the density of one point (default is 250)
    density_threshold : float, optional
        Minimum density of a resonance so that the resonance is not attenuated. If None, density_threshold is estimated
        by middle density of the spectrogram (default is None)
    density_factor : float, optional
        Factor that multiplies the middle density of the spectrogram to estimate the attenuation factor. Used when 
        denisty threshold is None (default is 5)

    Returns
    -------
    resonance3.ResonanceSpectrogram
        Resonance spectrogram denoised by density

    """

    density_spectrogram = np.hstack(
        discrete_density_spectrogram(spectrogram, slices_overlap=slices_overlap, freq_overlap=freq_overlap))

    if density_threshold == None:
        sorted_density = np.sort(np.hstack(density_spectrogram))
        density_threshold = density_factor * sorted_density[len(sorted_density) // 2]

    # Bigger the factor of the exp, the steeper
    new_elements = [
        Resonance(res.d if density > density_threshold else res.d * (np.exp(- 2 * (1 - density / density_threshold))),
                  res.w, res.z, res.N, res.sample_rate)
        for res, density in zip(spectrogram.elements, density_spectrogram)]

    ids = [hash(res) for res in new_elements]

    # CHANGED: was ResonanceSpectrogram at first
    return ResonanceSpectrogram(np.array(ids), np.array(spectrogram.onsets))


def denoise_by_dyns(spectrogram, min_overlap=1e-15, min_length=3, freq_ratio=20):
    """Function to denoise a spectrogram based its dynamic resonances.

    Parameters
    ----------

    spectrogram : resonance3.ResonanceSpectrogram
        Resonance spectrogram to denoise
    min_overlap : float, optional
        Min overlap between resonances to consider it as a match (default is 1e-15)
    min_length : int, optional
        Minimum length of a dynamic resonance to not to be filtered (default is 3)
    freq_ratio : float, optional
        Ratio of frequency in times/second relative to the original resonance for which candidates 
        are considered (default is 20)

    Returns
    -------
    resonance3.ResonanceSpectrogram
        Resonance spectrogram denoised by dynamic resonances

    """

    dyns = get_dynamic_resonances(spectrogram,
                                  min_overlap,
                                  overlap_type="exponential",
                                  freq_ratio=freq_ratio,
                                  windows=1,
                                  multiple_match=False,
                                  mode='forward',
                                  distance='residue_power')

    dyns = dyns.filter(lambda onset, dynamic: len(dynamic) >= min_length)

    return ResonanceSet(np.hstack(dynamic.elements for dynamic in dyns.elements),
                        np.hstack(dynamic.onsets + onset for dynamic, onset
                                  in zip(dyns.elements, dyns.onsets)))


def denoise(spectrogram, density_factor=1, power_factor=0.5, min_overlap=1e-15, min_length=4):
    """Function to denoise a spectrogram based its dynamic resonances.

    Parameters
    ----------

    spectrogram : resonance3.ResonanceSpectrogram
        Resonance spectrogram to denoise
    density_factor : float, optional
        Factor that multiplies the middle density of the spectrogram to estimate the attenuation factor. 
        If None, no density denoising is applied (default is 1)
    power_factor : float, optional
        Factor that multiplies the average power to estimate the minimum power thresohld. (default is 0.5)
        If None, no power denoising is applied
    min_overlap : float, optional
        Min overlap between resonances to consider it as a match (default is 1e-15)
        If None, no dynamic resonances denoising is applied
    min_length : int, optional
        Minimum length of a dynamic resonance to not to be filtered (default is 4)

    Returns
    -------
    resonance3.ResonanceSpectrogram
        Denoised resonance spectrogram
    """

    if density_factor is not None:
        spectrogram = denoise_by_density(spectrogram, slices_overlap=7, freq_overlap=250, density_factor=density_factor)

    if power_factor is not None:
        spectrogram = denoise_by_power(spectrogram, power_factor=power_factor)

    if min_overlap is not None:
        spectrogram = denoise_by_dyns(spectrogram, min_overlap=min_overlap, min_length=min_length, freq_ratio=10)

    return spectrogram


""" Additional utilities """


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """Function to print a progress bar

    Parameters
    ----------
    iteration : int
        Iteration number
    total : int
        Total number of iterations

    """

    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def maxIndex(nestedList):
    """Function to obtain the index of the maximum element in a nested list.

    Parameters
    ----------
    nestedList : List[List]
        Nested list of elements for which the index of the maximum element is obtained

    Returns
    -------
    Tuple
        Index of the maximum element
    """

    m = (0, 0)
    maximum = 0
    for i in range(len(nestedList)):
        for j in range(len(nestedList[i])):
            if nestedList[i][j] > maximum:
                m = (i, j)
                maximum = nestedList[i][j]
    return m
