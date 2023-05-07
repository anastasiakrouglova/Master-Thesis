using PlotlyJS, ClusterAnalysis, StatsBase, DataFrames, CSV, LinearAlgebra

using ScikitLearn
@sk_import cluster: (KMeans)

function remove_noise(data, min_power, min_frequency, max_frequency)
    # Noise in the onset appears at the beginning and end of the data from padding the signal 
    # with zeros on the left and right so that the window/step-size/overlap work out.
    data = data[(data.onset .!= 0) .& (data.onset .!= maximum(data.onset)), :]
    
    # good value (tested on flute-a4.wav) for min_power = 0.001
    data = data[(data.power .> min_power), :]

    # Remove frequencies above max_frequency (default 2000) and complex part
    data = data[(min_frequency .< data.frequency) .& (data.frequency .< max_frequency), :]
    
    return data
end


# TODO: Find epsilon and min_pts with hyperparameter tuning
# ϵ = 0.5; # min_pts = 5;
function findClusters(raw, ϵ, min_pts, min_power, min_frequency, max_frequency)
    # Denoise raw data: currently already done for f0 extraction
    df = remove_noise(raw, min_power, min_frequency, max_frequency)

    df[!,:onset_s] = (df.onset ./ df.sample_rate)
    normalize!(df.power, 2)


    # Convert data to a matrix
    # X = convert(Matrix, df[:,[1, 6, 8]]) # with power
    X = convert(Matrix, df[:,[1, 6]]) # ignore power in clustering

    # normalize matrix
    dt = fit(ZScoreTransform, X, dims=1)
    mat = StatsBase.transform(dt, X)

    # Run DBSCAN 
    m = dbscan(mat, ϵ, min_pts); #returns object dbscan!

    # Run Kmeans: just as a comparison
    # clusterer = KMeans(n_clusters=4, random_state=1, n_init=8)
    # cluster_labels = clusterer.fit_predict(X)

    # Put labels from clustering back to a dataframe
    # print(m.labels)
    df[!,:dynamicResonance] = m.labels

    # kmeans : just as a comparison
    # df[!,:dynamicResonance] = cluster_labels

    return df
end


function plotCluster(df)
    # https://plotly.com/julia/reference/scatter3d/
    p = plot(
        df, 
        Layout(scene = attr(
                        xaxis_title="Time (s)",
                        yaxis_title="Frequency (Hz)",
                        zaxis_title="Power"),
                        #margin=attr(r=100, b=150, l=50, t=50)
                        ),
        x=:onset_s, 
        y=:frequency, z=:power, color=:dynamicResonance,  
        type="scatter3d", mode="markers", 
        marker_size=2
    )

    name = "Clustering of resonances"
    # Default parameters which are used when `layout.scene.camera` is not provided
    camera = attr(
        up=attr(x=0, y=0, z=1),
        center=attr(x=0, y=0, z=0),
        eye=attr(x=-1.25, y=-1.25, z=1.25)
    )
    relayout!(p, scene_camera=camera, title=name)
    p
end

raw = DataFrame(CSV.File("./fpt/data/output/flute-a4.csv"))
raw[!,:id] = collect(1:size(raw)[1])
df = findClusters(raw, 0.4, 4, 0.001, 0, 2000)
CSV.write("./fpt/data/output/filtered-clustered-flute_a4.csv", df)


plotCluster(df) 







# raw = DataFrame(CSV.File("./fpt/data/output/flute-d4.csv"))
# raw = DataFrame(CSV.File("./fpt/data/output/nine_N500.csv"))
# raw = DataFrame(CSV.File("./fpt/data/output/piano-chords_83_C.csv"))
# raw = DataFrame(CSV.File("./fpt/data/output/HALFTONES/flute_C5F.csv"))


# raw = DataFrame(CSV.File("./fpt/data/output/nine_N500.csv"))
# raw = DataFrame(CSV.File("./fpt/data/output/A_maj_4_0.csv"))


# check ad different frequencies and compare
#df = findClusters(raw, 0.5, 5, 0.0001, 0, 2000) # flute d4: until 4000 frequency
#df = findClusters(raw, 0.3, 8, 0.005, 0, 4000) # Piano 
# for speach: min freq 100
# https://flypaper.soundfly.com/produce/eqing-vocals-whats-happening-in-each-frequency-range-in-the-human-voice/
#df = findClusters(raw, 0.6, 8, 0.00000001, 60, 1000) # Nine (freq we can hear)
#df = findClusters(raw, 0.15, 5, 0.0001, 0, 1000) # flute afaf
#df = findClusters(raw, 0.12, 5, 0.001, 0, 1000) # piano afaf
#df = findClusters(raw, 0.034, 5, 0.0001, 0, 4000) # flute afaf


