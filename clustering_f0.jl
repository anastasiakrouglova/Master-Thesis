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

filename = "flute_syrinx_1_f0"
PATH = "./fpt/data/output/scores/" * filename * ".csv"

function __main__(path, accuracy)
    raw = DataFrame(CSV.File(path))
    # Additional id column for hierarchical knowledge representation
    raw[!,:id] = collect(1:size(raw)[1])

    clustered_df = hyperparameterTuning(raw, accuracy)

    # Export clustered data
    CSV.write("./fpt/data/output/filtered-clustered-C5F4_f0.csv", clustered_df)

    # Plot the Distances
    # plot(scatter(df_distance, x=:index, y=:distance, mode="markers"))
    # Plot the Clusters
    plotCluster(clustered_df) 
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
        eye=attr(x=-1.55, y=-1.55, z=1.55)
    )
    relayout!(p, scene_camera=camera, title=name)

    savefig(p, "test.png")

    p
end

function featureNormalization(df)
    data = DataFrame(onset=df.onset, frequency=df.frequency)
    mapper = DataFrameMapper([([:onset], StandardScaler()),
                            ([:frequency], StandardScaler())]);
    mapper = fit_transform!(mapper, copy(data))
end


function hyperparameterTuning(df, accuracy)
    # no denoise needed
    df[!,:onset_s] = (df.onset ./ df.sample_rate)
    normalize!(df.power, 2)
    # Convert data to a normalized matrix
    X = featureNormalization(df) #convert(Matrix, df[:,[2, 7]]) # ignore power in clustering
    knee_eps = knee_epsilonTuning(X)

    # Normalize matrix
    # dt = fit(ZScoreTransform, X, dims=1)
    # mat = StatsBase.transform(dt, X)
    #normOnFreq = onFreqNormalization(df)

    max_silouette = 0
    best_pts = 0
    best_eps = 0
    
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
    println("knee method eps:", knee_eps) 
    println("silhoutte pts:", best_pts)
    println("silhoutte eps:", best_eps)
    println("--------------")

    # return  best min_pts
    best_clustering = dbscan(X, best_eps, best_pts); 

    df[!,:dynamicResonance] = best_clustering.labels

    # CONCLUSION: KNEE METHOD DOES NOT WORK FOR OUR PROBLEM!!!
    # reason: not in combination of 2 parameters:
    return df
end

# Experimental setup, did not give appropriate results.
function knee_epsilonTuning(X)
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

# accuracy must be a value between 10 and 50, since 0.1 <= eps <= 0.5
# The higher the value, the less accuracy (just inverse for user later)
__main__(PATH, 10)



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

#df = findClusters(raw, 0.10, 23) # halftones
#df = findClusters(raw, 0.07, 18) # syrinx

