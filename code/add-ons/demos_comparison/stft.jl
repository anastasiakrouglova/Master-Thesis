using DSP, WAV, PlotlyJS, DataFrames, ClusterAnalysis  # "using" makes the listed modules available for the
                       # user, like "import" in other languages

using StatsBase
using PyCall
using Conda
using Statistics, Distributions
using ScikitLearn
np = pyimport("numpy")
ENV["PYTHON"]=""
push!(pyimport("sys")."path", "/Users/nastysushi/Mirror/_MULTIMEDIA/THESIS/thesis/github/code/")
push!(pyimport("sys")."path", "./") # add a hardcoded path if it doesn't work: push!(pyimport("sys")."path", "<PATH>/code/")
kneed = pyimport("kneed")
# import libraries
@sk_import preprocessing: (StandardScaler)
@sk_import metrics: (silhouette_samples, silhouette_score)


                       
function remove_noise(data, min_power, max_frequency)
    # Noise in the onset appears at the beginning and end of the data from padding the signal 
    # with zeros on the left and right so that the window/step-size/overlap work out.
    data = data[(data.onset .!= 0) .& (data.onset .!= maximum(data.onset)), :]
    
    # good value (tested on flute-a4.wav) for min_power = 0.001
    minPow = 0.000001
    data = data[(data.power .> minPow), :]

    # Remove frequencies above 2000 and complex part
    data = data[(0 .< data.frequency) .& (data.frequency .< max_frequency), :]
    # normalize!(data.power, 2)

    return data
end


# # Loading and plotting an audio signal
s, fs = wavread("code/fpt/data/input/polyphonic/K331-Tri_short.wav")
# s, fs = wavread("./fpt/data/input/scores/flute_syrinx_2.wav")

# PROBLEM: JUMPS OF 661 HERE, at Flute: 512

# Amount of samples

S = spectrogram(s[:,1], 512*2, 0; # round(Int, 25e-3*fs),round(Int, 10e-3*fs);
 window=hanning
 )

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
    ACCURACY = 6

    best_pts, best_eps = silhouetteScore(X, ACCURACY)
    m = dbscan(mat, best_eps, best_pts); #returns object dbscan!

    # Put labels from clustering back to a dataframe
    # print(m.labels)
    df[!,:cluster] = m.labels

    return df
end

function plotCluster2D(df)
    plot(
        df, x=:onset, y=:frequency, #color=:species,
        kind="scatter", mode="markers", marker_size=2,
        labels=Dict(
            :onset => "Onset (s)",
            :frequency => "Frequency (Hz)",
            #:species => "Species"
        ),
        Layout(title="Fourier Spectrogram",)
    )
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
        x=:onset, 
        y=:frequency,
        z=:power, 
        color=:cluster,
        type="scatter3d", 
        mode="markers", 
        marker_size=2
    )

    name = "Fast fourier transform"
    # Default parameters which are used when `layout.scene.camera` is not provided
    camera = attr(
        up=attr(x=0, y=0, z=1),
        center=attr(x=0, y=0, z=0),
        eye=attr(x=-2.40, y=0, z=1.25)
    )

    relayout!(p, scene_camera=camera, title=name)
    p
end


function silhouetteScore(X, accuracy)
    max_silouette = 0
    best_pts = 0
    best_eps = 0

    # knee_eps = knee_epsilonTuning(X)
    
    for min_pts in 3:20 
        for eps in range(0.01, step=0.01, length=accuracy) # TODO: user can adjust accuracy of the algorithm to have more or less notes found!! length is the parameter that will be adjusted in this case
        # Run DBSCAN 
            m = dbscan(X, eps, min_pts); #returns object dbscan!
            # Put labels from clustering back to a dataframe
            cluster_labels = m.labels

            # Ignore tuning where all resonances are labeled as noise
            if (!all(y->y==cluster_labels[1],cluster_labels))
                # Metric for the evaluation of the quality of a clustering technique
                silhouette_avg = silhouette_score(X, cluster_labels)
                # println("for min_pts=", min_pts, "and eps", eps, "the average silhouette_score is :", silhouette_avg)
                if (silhouette_avg > max_silouette)
                    max_silouette = silhouette_avg
                    best_pts = min_pts
                    best_eps = eps
                end
            end
        end
    end


    
    println("--------------")
    # println("knee method eps:", knee_eps) 
    println("silhoutte pts:", best_pts)
    println("silhoutte eps:", best_eps)
    println("--------------")

    return best_pts, best_eps

end


df_cluster = findClusters(df, 0.01, 5, 0.1, 2000)

# plotCluster2D(df_cluster)

plotCluster(df_cluster)
