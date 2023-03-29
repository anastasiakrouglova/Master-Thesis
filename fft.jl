# using FourierAnalysis, PlotlyJS

# using SignalAnalysis


using DSP, WAV, PlotlyJS, DataFrames  # "using" makes the listed modules available for the
                       # user, like "import" in other languages


                       
function remove_noise(data, min_power, max_frequency)
    # Noise in the onset appears at the beginning and end of the data from padding the signal 
    # with zeros on the left and right so that the window/step-size/overlap work out.
    data = data[(data.time .!= 0) .& (data.time .!= maximum(data.time)), :]
    
    # good value (tested on flute-a4.wav) for min_power = 0.001
    data = data[(data.power .> min_power), :]

    # Remove frequencies above 2000 and complex part
    data = data[(0 .< data.frequency) .& (data.frequency .< max_frequency), :]
    
    return data
end


# # Loading and plotting an audio signal
s, fs = wavread("./fpt/data/input/flute-a4.wav")

# PROBLEM: JUMPS OF 661 HERE, at Flute: 512

S = spectrogram(s[:,1], round(Int, 25e-3*fs),
round(Int, 10e-3*fs); window=hanning)

t = time(S) / fs
# In digital signal processing (DSP), a normalized frequency is a ratio of a variable frequency (f) and a constant frequency associated with a system (such as a sampling rate, fs)
f = freq(S) * fs # something wrong: is between 0 and 0.5? ah in Hz, moet
p = power(S)
p = vec(p)

x = repeat(t, inner=length(f))
y = repeat(f, outer=length(t)),
println("-----")
println(length(x))
println(length(y[1]))
println(length(p))
println("-----")
print(maximum(f))

df = DataFrame(time=x, frequency=y[1], power=p)
# print(df.time)
df = remove_noise(df, 0.1, 2000)


function plotCluster(df)
    # https://plotly.com/julia/reference/scatter3d/
    p = plot(
        df, 
        Layout(scene = attr(
                        xaxis_title="Time (s)",
                        yaxis_title="Frequency (Hz)",
                        zaxis_title="Power (dB)",
                        )
                        #margin=attr(r=50, b=50, l=50, t=50)
                        ),
        x=:time, y=:frequency, z=:power,
        type="scatter3d", mode="markers", 
        marker_size=3
    )

    name = "Fast fourier transform"
    # Default parameters which are used when `layout.scene.camera` is not provided
    camera = attr(
        up=attr(x=0, y=0, z=1),
        center=attr(x=0, y=0, z=0),
        eye=attr(x=-1.25, y=-1.25, z=1.25)
    )

    relayout!(p, scene_camera=camera, title=name)
    p
end



plotCluster(df)

# Create 3D plot
# plot(
#     x = repeat(t, inner=length(f)),
#     y = repeat(f, outer=length(t)),
#     z = p[:],
#     type = "surface",
#     colorscale = "Viridis",
#     Layout(scene = attr(
#         xaxis_title="Time (s)",
#         yaxis_title="Frequency (Hz)",
#         zaxis_title="Power (dB)"),
#         #margin=attr(r=50, b=50, l=50, t=50)
#         ),
# )



# window_length = round(Int, 25e-3*fs)

# Load audio file
#filename = "audio.wav"
# (x, fs) = wavread("./fpt/data/input/flute-a4.wav")

# # Compute spectrogram
# window_length = round(Int, 25e-3 * fs)
# segment_length = round(Int, 10e-3 * fs)
# S, f, t = spectrogram(x, DSP.Windows.(hanning(window_length)), segment_length, fs=fs)

# # Compute power spectral density
# P = abs2.(S) ./ window_length

# println("----\n")
# #println(t)
# println("----\n")


# window_length = round(Int, 25e-3 * fs)
# segment_length = round(Int, 10e-3 * fs)

# S, f, t = spectrogram(s[:,1], window(hanning, window_length), segment_length, fs=fs)
# p = abs2.(S) ./ window_length


# S, f, t = spectrogram(s[:,1], round(Int, 25e-3*fs), round(Int, 10e-3*fs), fs=fs)
# p = abs2.(S) ./ size(S, 1)

# S, f = spectrogram(s[:,1], window=hanning, nfft=round(Int, 25e-3*fs), noverlap=round(Int, 10e-3*fs))
# t = (0:size(S, 2)-1) .* (size(S, 2) > 1) .* (size(S, 2)-1) * (nfft - noverlap) / fs .+ nfft / (2fs)
# p = abs2.(S) ./ (nfft * sum(window .^ 2) / noverlap)


# p = vec(p) # because pauwer returns otherwise array



# print(length(p))



# Create a dataframe from time, frequency and power
# df = DataFrame(C=p)


# function plotCluster(df)
#     # https://plotly.com/julia/reference/scatter3d/
#     plot(
#         df, 
#         Layout(scene = attr(
#                         xaxis_title="Time",
#                         yaxis_title="Frequency",
#                         zaxis_title="Power"),
#                         #margin=attr(r=50, b=50, l=50, t=50)
#                         ),
#         x=:onset, y=:frequency, z=:power, color=:cluster,  
#         type="scatter3d", mode="markers", 
#         marker_size=3
#     )
# end
























