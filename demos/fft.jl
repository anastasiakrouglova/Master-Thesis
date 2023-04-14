# using FourierAnalysis, PlotlyJS

# using SignalAnalysis


using DSP, WAV, PlotlyJS, DataFrames, ClusterAnalysis  # "using" makes the listed modules available for the
                       # user, like "import" in other languages


                       
function remove_noise(data, min_power, max_frequency)
    # Noise in the onset appears at the beginning and end of the data from padding the signal 
    # with zeros on the left and right so that the window/step-size/overlap work out.
    data = data[(data.onset .!= 0) .& (data.onset .!= maximum(data.onset)), :]
    
    # good value (tested on flute-a4.wav) for min_power = 0.001
    data = data[(data.power .> min_power), :]

    

    # Remove frequencies above 2000 and complex part
    data = data[(0 .< data.frequency) .& (data.frequency .< max_frequency), :]
    
    normalize!(data.power, 2)

    return data
end


# # Loading and plotting an audio signal
s, fs = wavread("./fpt/data/input/flute-a4.wav")

# PROBLEM: JUMPS OF 661 HERE, at Flute: 512

S = spectrogram(s[:,1], 512, 0; # round(Int, 25e-3*fs),round(Int, 10e-3*fs);
 window=hanning)

t = (time(S) ./ fs) * 44100
# In digital signal processing (DSP), a normalized frequency is a ratio of a variable frequency (f) and a constant frequency associated with a system (such as a sampling rate, fs)
f = freq(S) * fs 
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

df = DataFrame(onset=x, frequency=y[1], power=p)
# print(df.time)
# df = remove_noise(df, 0.1, 2000)

function findClusters(raw, ϵ, min_pts, min_power, max_frequency)
    # Denoise raw data
    df = remove_noise(raw, min_power, max_frequency)

    df.onset = (df.onset ./ 44100)
    df.frequency = df.frequency

    # Convert data to a matrix
    X = convert(Matrix, df[:,[1, 2, 3]])

    # normalize matrix
    dt = fit(ZScoreTransform, X, dims=1)
    mat = StatsBase.transform(dt, X)

    # Run DBSCAN 
    m = dbscan(mat, ϵ, min_pts); #returns object dbscan!

    # Put labels from clustering back to a dataframe
    # print(m.labels)
    df[!,:cluster] = m.labels

    return df
end


function plotCluster(df)
    # https://plotly.com/julia/reference/scatter3d/
    p = plot(
        df, 
        Layout(scene = attr(
                        xaxis_title="Time (s)",
                        yaxis_title="Frequency (Hz)",
                        zaxis_title="Power",
                        )
                        #margin=attr(r=50, b=50, l=50, t=50)
                        ),
        x=:onset, y=:frequency, z=:power, #color=:cluster,
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


df_cluster = findClusters(df, 0.5, 5, 0.1, 2000)
plotCluster(df_cluster)
