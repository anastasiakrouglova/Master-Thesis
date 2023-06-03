using WAV 
using Plots #PlotlyJS
using FFTW
using SignalAnalysis

snd, sampFreq = wavread("fpt/data/input/scores/flute_syrinx_2.wav")

s1 = snd[:,1]

# Plot time domain
# timeArray = (0:(1292-1)) / sampFreq
# timeArray = timeArray * 1000 #scale to milliseconds

timePlot = plot(s1, label=["Time" "Amplitude"])
# freqPlot = psd(s1)
spec = specgram(s1)



# Plot frequency domain
# n = length(s1)
# p = fft(s1)

# nUniquePts = ceil(Int, (n+1)/2)
# p = p[1:nUniquePts]
# p = broadcast(abs, p)


# p = p ./ n #scale
# p = p.^2  # square it
# # odd nfft excludes Nyquist point
# if n % 2 > 0
#     p[2:length(p)] = p[2:length(p)]*2 # we've got odd number of   points fft
# else 
#     p[2: (length(p) -1)] = p[2: (length(p) -1)]*2 # we've got even number of points fft
# end



# freqArray = (0:(nUniquePts-1)) * (sampFreq / n)
# plot(scatter(;x=freqArray/1000, y=10*broadcast(log10, p)),
#      Layout(xaxis_title="Frequency (kHz)",
#             xaxis_zeroline=false,
#             xaxis_showline=true,
#             xaxis_mirror=true,
#             yaxis_title="Power (dB)",
#             yaxis_zeroline=false,
#             yaxis_showline=true,
#             yaxis_mirror=true))