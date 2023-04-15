using PlotlyJS, ClusterAnalysis, StatsBase, DataFrames, CSV, LinearAlgebra


function remove_noise(data, min_power, max_frequency)
    # Noise in the onset appears at the beginning and end of the data from padding the signal 
    # with zeros on the left and right so that the window/step-size/overlap work out.
    data = data[(data.onset .!= 0) .& (data.onset .!= maximum(data.onset)), :]
    
    # good value (tested on flute-a4.wav) for min_power = 0.001
    data = data[(data.power .> min_power), :]

    # Remove frequencies above 2000 and complex part
    data = data[(0 .< data.frequency) .& (data.frequency .< max_frequency), :]
    
    return data
end


# TODO: Find epsilon and min_pts with hyperparameter tuning
# ϵ = 0.5; # min_pts = 5;
function findClusters(raw, ϵ, min_pts, min_power, max_frequency)
    # Denoise raw data
    df = remove_noise(raw, min_power, max_frequency)

    df[!,:onset_s] = (df.onset ./ df.sample_rate)
    normalize!(df.power, 2)

    # Convert data to a matrix
    X = convert(Matrix, df[:,[1, 6, 8]])

    # normalize matrix
    dt = fit(ZScoreTransform, X, dims=1)
    mat = StatsBase.transform(dt, X)

    # Run DBSCAN 
    m = dbscan(mat, ϵ, min_pts); #returns object dbscan!

    # Put labels from clustering back to a dataframe
    # print(m.labels)
    df[!,:dynamicResonance] = m.labels

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
        marker_size=3
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
# raw = DataFrame(CSV.File("./fpt/data/output/nine_N500.csv"))
# raw = DataFrame(CSV.File("./fpt/data/output/A_maj_4_0.csv"))
df = findClusters(raw, 0.5, 5, 0.001, 2000)

CSV.write("./fpt/data/output/filtered-clustered-flute_a4.csv", df)


plotCluster(df)