# NOTE: f_0 has the additional row "ID", so code slightly different
using PlotlyJS, ClusterAnalysis, StatsBase, DataFrames, CSV, LinearAlgebra
using PyCall
using Conda
using ScikitLearn
np = pyimport("numpy")
ENV["PYTHON"]=""
# using Pkg
# Pkg.build("PyCall")
push!(pyimport("sys")."path", "/Users/nastysushi/Mirror/_MULTIMEDIA/THESIS/thesis/github/")
kneed = pyimport("kneed")
# import libraries
@sk_import preprocessing: (StandardScaler)
@sk_import metrics: (silhouette_samples, silhouette_score)
@sk_import cluster: (KMeans)

PATH = "./fpt/data/output/filtered-clustered-C5F4_f0.csv"
PATH = "./fpt/data/output/syrinx_f0.csv"

function __main__(path)
    raw = DataFrame(CSV.File(path))
    # Additional id column for hierarchical knowledge representation
    raw[!,:id] = collect(1:size(raw)[1])
    X = dataNormalization(raw)

    # hyperparameters
    #MIN_PTS = min_ptsTuning(X) # number of dimensions * 2
    #EPSILON = epsilonTuning(X)

    #println(MIN_PTS)
    #println(EPSILON)

    #df = findClusters(raw, 0.10, 23) # halftones
    #df = findClusters(raw, 0.07, 18) # syrinx
    df = findClusters(raw, 0.07, 6)

    # Export clustered data
    CSV.write("./fpt/data/output/filtered-clustered-C5F4_f0.csv", df)

    # Plot the Distances
    # plot(scatter(df_distance, x=:index, y=:distance, mode="markers"))
    # Plot the Clusters
    plotCluster(df) 
end


function findClusters(df, ϵ, min_pts)
    # no denoise needed
    df[!,:onset_s] = (df.onset ./ df.sample_rate)
    normalize!(df.power, 2)
    # Convert data to a matrix
    X = convert(Matrix, df[:,[2, 7]]) # ignore power in clustering

    # Normalize matrix
    dt = fit(ZScoreTransform, X, dims=1)
    mat = StatsBase.transform(dt, X)

    # Run DBSCAN 
    m = dbscan(mat, ϵ, min_pts); #returns object dbscan!
    # Put labels from clustering back to a dataframe
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

function dataNormalization(df)
    data = DataFrame(onset=df.onset, frequency=df.frequency)
    mapper = DataFrameMapper([([:onset], StandardScaler()),
                            ([:frequency], StandardScaler())]);
    mapper = fit_transform!(mapper, copy(data))
end

function min_ptsTuning(X)
    max_silouette = 0
    best_cluster = 0

    for n_clusters in 3:50
        clusterer = KMeans(n_clusters=n_clusters, random_state=10, n_init=10)
        cluster_labels = clusterer.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)

        # println("for n_clusters=", n_clusters, "The average silhouette_score is :", silhouette_avg)
        if (silhouette_avg > max_silouette)
            max_silouette = silhouette_avg
            best_cluster = n_clusters
        end

        # compute the silhouette score for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)
    end

    best_cluster
end

function epsilonTuning(X)
    df_distance = DataFrame([[],[]], ["index", "distance", ])

    l_X = size(X, 1)-1
    for i in 1:l_X
        dist = np.linalg.norm(X[i, :]-X[i+1, :])
        push!(df_distance, [string(i), dist])
    end 
    df_distance = sort!(df_distance, :distance)

    # Knee extraction, Satopaa 2011
    distances = df_distance.distance
    i = 1:length(distances)
    knee = kneed.KneeLocator(i, distances, S=1, curve="convex", direction="increasing", interp_method="polynomial")
    # Returns the epsilon
    distances[knee.knee]
end

__main__(PATH)


########################## TODO #######################################

# SPEED TEST:
# TODO: @time all functions (20 times and take average), do the same with Dynamic Resonances: show which one is faster 
#@time min_ptsTuning(X)

# compilation
# @profview plotCluster(df)
# # pure runtime
# @profview plotCluster(df)

# STATISTICAL COMPARISON
# TODO: run on 20 pieces: say what the accuracy is met hyperparameter tuning
# Accuracy met manuele hyperparameter tuning: 100%
# automatische hyperparameter tuning met kleinste afstand: ...



