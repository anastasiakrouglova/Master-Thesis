# NOTE: f_0 has the additional row "ID", so code slightly different

using PlotlyJS, ClusterAnalysis, StatsBase, DataFrames, CSV, LinearAlgebra
using PyCall
using Conda
using ScikitLearn

using NearestNeighbors
using PyCall
np = pyimport("numpy")

# nn = pyimport("sklearn")
#from sklearn.neighbors import NearestNeighbors

using NearestNeighbors

# TODO: Find epsilon and min_pts with hyperparameter tuning
# ϵ = 0.5; # min_pts = 5;
function findClusters(df, ϵ, min_pts)
    # no denoise needed
    df[!,:onset_s] = (df.onset ./ df.sample_rate)
    normalize!(df.power, 2)
    # Convert data to a matrix
    X = convert(Matrix, df[:,[2, 7]]) # ignore power in clustering

    #hyperparameterTuning(X)

    # Normalize matrix
    dt = fit(ZScoreTransform, X, dims=1)
    mat = StatsBase.transform(dt, X)


    range_eps = [0.1, 0.2, 0.3, 0.4, 0.5]
    max_silhouette_avg = [0, 0]
    for i in 1:length(range_eps)
        println("Eps value is", i)
        db = dbscan(mat, ϵ, min_pts);
        print(typeof(db))
        #core_samples_mask = np.zeros_like(db.indices)

        #core_samples_mask[db.index] = True
        # labels = db.labels_
        # print(set(labels))
        # print(set(labels))

        silhouette_avg = silhouette_score(mat, labels)
        
        if(max_silhouette_avg[0] < silhouette_avg)
            max_silhouette_avg[0] = silhouette_avg
            max_silhouette_avg[1] = i
        end
           
        # print("For eps value ="+str(i), #labels,
        #       "The average silhouette_score is :", silhouette_avg)
    end
    # print("BEST EPS IS", max_silhouette_avg[1] )
    # max_silhouette_avg[1]
        

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

# Hyperparameter tuning EPS
#min_samples = dim*2

function hyperparameterTuning(X)
    dim = ndims(X)

    # X = X[1:10]

    
    # for i in 1:length(X)
    #     for j in 1:length(X)
    #         # find distance from i to j
    #         dist = np.linalg.norm(X[i, :]-X[j, :]) # works, is just slow
    #         print(dist)
    #         # find min values of distances to nearest 3
    #         min()
    #     end
    # end

    # sort distances in ascending and plot tof ind each value

    # # elbatta, 2012 uitleggen in verslag
    k = 4 #dim*2 # if 1 or 2, subject to noise; not goed, so altijd dim*2
    points = rand(1397, 6)

    # print(X[:,2])

    kdtree = KDTree(X)
    println(kdtree)

    idxs, dists = knn(kdtree, points, k, true)
    
    dists = np.sort(dists, axis=0) # very slow calculation but works

    println(idxs)
    println(dists)

end


MIN_PTS = 4 # number of dimensions * 2

raw = DataFrame(CSV.File("./fpt/data/output/C5F_f0.csv"))
raw[!,:id] = collect(1:size(raw)[1])
#df = findClusters(raw, 0.04, 5) # halftones
df = findClusters(raw, 0.06, MIN_PTS) # halftones

#CSV.write("./fpt/data/output/filtered-clustered-flute_a4.csv", df)
CSV.write("./fpt/data/output/filtered-clustered-C5F4_f0.csv", df)

# plotCluster(df) 

# plot(dists)




# TODO: run on 20 pieces: say what the accuracy is met hyperparameter tuning
# Accuracy met manuele hyperparameter tuning: 100%
# automatische hyperparameter tuning met kleinste afstand: ...