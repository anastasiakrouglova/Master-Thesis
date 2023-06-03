# This file runs the Fast Padé transform (FPT), 
# extract the fundamental and saves both files as a CSV in `fpt/data/output` together with the DRS

# Run the clustering algorithm written in Julia on the saved CSV's for the c_<file>.csv 
# and test the hierarchical knowledge representation in knowledge_heararchy.ipynb

#import os, sys
#from matplotlib import pyplot as plt

## 0. Import the library of functions
#sys.path.insert(0, './fpt')
#from utilities_store import *
#import serialization as ser
using PyCall
ENV["PYTHON"]=""
using Pkg
Pkg.build("PyCall") # Rerun this if something is not working
push!(pyimport("sys")."path", "/Users/nastysushi/Mirror/_MULTIMEDIA/THESIS/thesis/github/fpt/")

us = pyimport("utilities_store") # to check what is inside: PyCall.inspect[:getmembers](us)


# plot spectrogram
plt = pyimport("matplotlib.pyplot")
x = range(0;stop=2*pi,length=1000); y = sin.(3*x + 4*cos.(2*x));
plt.plot(x, y, color="red", linewidth=2.0, linestyle="--")
plt.show()


## 1. Obtain spectrogram of the signal
# spectrogram, signal = get_spectrogram("fpt/data/input/scores/solo/natural/flute_syrinx/syrinx_1/syrinx_1.wav", N = 500, step_size = 250, 
#                                            power_threshold = 1e-9, amp_threshold = 1e-8)

## 2. Save the spectrogram as a csv file as png
# plot_spectrogram(spectrogram, max_freq = 2000, scale=100)
# plt.savefig('foo.png')

# ser.to_csv(spectrogram)

## 3. Fundamental frequency estimation
## Obtain the fundamental frequency of a signal and plot it in the orginal resonance spectrogram.

