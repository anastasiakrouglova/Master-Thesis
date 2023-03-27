using PlotlyJS, ClusterAnalysis, StatsBase, DataFrames, CSV


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
function findClusters(raw, ϵ, min_pts)
    # Parameters of DBSCAN


    # Denoise raw data
    df = remove_noise(raw, 0.001, 2000)

    # Convert data to a matrix
    X = convert(Matrix, df[:,[1, 6, 8]])

    # normalize matrix
    dt = fit(ZScoreTransform, X, dims=1)
    mat = StatsBase.transform(dt, X)

    # Run DBSCAN 
    m = dbscan(mat, ϵ, min_pts); #returns object dbscan!

    #m = kmeans(mat, 4); #returns object dbscan!

    # Put labels from clustering back to a dataframe
    # print(m.labels)
    df[!,:cluster] = m.labels

    # for k-means experiment
    # print(m.cluster)
    # df[!,:clusterK] = m.cluster
    return df
end


function plotCluster(df)
    # https://plotly.com/julia/reference/scatter3d/
    plot(
        df, 
        Layout(scene = attr(
                        xaxis_title="Onset",
                        yaxis_title="Frequency",
                        zaxis_title="Power"),
                        #margin=attr(r=50, b=50, l=50, t=50)
                        ),
        x=:onset, y=:frequency, z=:power, color=:cluster,  
        type="scatter3d", mode="markers", 
        marker_size=3
    )
end



# raw = DataFrame(CSV.File("./fpt/data/output/flute-a4.csv"))
# raw = DataFrame(CSV.File("./fpt/data/output/nine_N500.csv"))
# raw = DataFrame(CSV.File("./fpt/data/output/A_maj_4_0.csv"))
# df = findClusters(raw, 0.5, 5)
# plotCluster(df)