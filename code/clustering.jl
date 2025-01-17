# NOTE: f0 has the additional row "ID", so code slightly different
using PlotlyJS, ClusterAnalysis, StatsBase, DataFrames, CSV, LinearAlgebra
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
@sk_import cluster: (KMeans)

# filename = "flute_syrinx_3"
filename = "K331-Tri_short"

# PATH = "./code/fpt/data/output/scores/" * filename * ".csv"
# PATH_OUTPUT = "./code/fpt/data/output/scores/clustered/" * filename * ".csv"
# PATH_PNG = "./code/fpt/data/output/scores/" * filename * ".png"

PATH = "./code/fpt/data/output/polyphonic/" * filename * ".csv"
PATH_OUTPUT = "./code/fpt/data/output/polyphonic/" * filename * "_clustered.csv"
PATH_PNG = "./code/fpt/data/output/polyphonic/" * filename * ".png"


EPS = 0.05
PTS = 4


function main(path, accuracy)
    raw = DataFrame(CSV.File(path))

    # Additional id column for hierarchical knowledge representation
    raw[!,:id] = collect(1:size(raw)[1])

    # remove the negative resonances to perform machine learning techniques only on the real part 
    pos_raw = filter(:frequency => x -> x > 0, raw)

    # cluster the f0 subset
    #f0_raw = pos_raw[isequal.(pos_raw.f0,1), :]
    #f0_raw = filter(:f0 => isequal(1), f0_raw)
    f0_raw = filter(:power => x -> x > 0.00001, pos_raw)
    clustered_f0 = hyperparameterTuning(f0_raw, accuracy)

    for (id, f0) in zip(clustered_f0.id, clustered_f0.f0)
        indices = findall(x -> x == id, pos_raw.id)
        pos_raw[indices, :f0] .= f0
    end

    pos_raw[!, :harmonic] .= -1
    pos_raw[!, :likeliness] .= 0.5

    temp = pos_raw
    max = maximum(clustered_f0.f0)

    # Power denoiser
    for i in 1:max
        avg_power, n_elements = avg_power_cluster(pos_raw, i)

        println("avg power: ", avg_power)
        println("n_elements: ", n_elements)

        if (avg_power < 0.0001 || n_elements < 10)
            temp = temp[temp.f0 .!= i, :]
        end
    end

    for i in 1:maximum(clustered_f0.f0)
        # Harmonics
        pos_raw = overtoneSlice(pos_raw, i)
    end


    CSV.write(PATH_OUTPUT, temp)
    lim_pos_raw = temp[temp.likeliness .<= 1, :]
    overtones_limFreq = lim_pos_raw[lim_pos_raw.frequency .<= 2000, :]
    filter_nonf0 = overtones_limFreq[overtones_limFreq.f0 .!= 0, :]
    
    #plotharmonic(overtones_limFreq) 
    plotf0(filter_nonf0) 
end

function avg_power_cluster(df, i)
    cluster = df[df.f0 .== i, :]
    vec_cluster = collect(cluster.power)
    avg_cluster = mean(vec_cluster)
    #avg_power = round(avg_cluster, digits = 2)
    println(avg_cluster)

    n_elements = nrow(cluster)

    return avg_cluster, n_elements
end

function overtoneSlice(df, i)
    f0_value = getf0(df, i)
    f0_resonances = df[df.f0 .== i, :]
    
    min = minimum(f0_resonances.onset)
    max = maximum(f0_resonances.onset)

    println("Cluster ", i, ", min: ",min, ", max: ",max)

    note_slice = findall(x -> min <= x <= max, df.onset)
    df[note_slice, :likeliness] .= (df[note_slice, :].frequency ./ f0_value) .% 1
        
    # Define the parameters for the Gaussian function
    mu = 0.5  # Mean
    sigma = 0.5  # Standard deviation

    # Calculate the Gaussian function values
    gaussian_values = pdf(Normal(mu, sigma), df[note_slice, :likeliness])

    # Multiply the likeliness column with the Gaussian values
    df[note_slice, :likeliness] .= df[note_slice, :likeliness] .* gaussian_values

    # Classify overtones with same id as f0
    likeliness = intersect(findall(x -> min <= x <= max, df.onset), findall(x -> x <= 0.01, df.likeliness))
    df[likeliness, :harmonic] .= i

    return df
end

function getf0(df, i)
    cluster = df[df.f0 .== i, :]
    vec_cluster = collect(cluster.frequency)
    avg_cluster = mean(vec_cluster)
    frequency_pred = round(avg_cluster, digits = 2)
    println(frequency_pred)

    return frequency_pred
end


# Euclidean distance onset/frequency
function featureNormalization(df)
    data = DataFrame(onset=df.onset, frequency=df.frequency)
    mapper = DataFrameMapper([([:onset], StandardScaler()),
                            ([:frequency], StandardScaler())
                            ]);
    mapper = fit_transform!(mapper, copy(data))
end


function silhouetteScore(X, accuracy)
    max_silouette = 0
    best_pts = 0
    best_eps = 0

    knee_eps = knee_epsilonTuning(X)
    
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

    return best_pts, best_eps, knee_eps

end

function hyperparameterTuning(df, accuracy)
    # no denoise needed
    df[!,:onset_s] = (df.onset ./ df.sample_rate)
    normalize!(df.power, 2)
    # Convert data to a normalized matrix
    df[!,:similarity] = resonanceSimilarity(df)

    X = featureNormalization(df)

    # Calculate Silhouette score and knee
    best_pts, best_eps, knee_eps = silhouetteScore(X, accuracy)

    best_clustering = dbscan(X, best_eps, best_pts); # Best: eps = 0.06, pts = 9
    
    df[!,:f0] = best_clustering.labels


    return df
end


############################## EXPERIMENTS WITH OTHER NORMALIZATION METHODS #############################
# Test: Euclidean distance between amplitude/decay functions

function onsetNormalization(df)
    data = DataFrame(onset=df.onset)
    mapper = DataFrameMapper([([:onset], StandardScaler())
                            ]);
    mapper = fit_transform!(mapper, copy(data))
end


function similarityNormalization(df)

    formatted_d = map(x -> replace(x, 'j' => "im", '(' => "", ')' => ""), df.d)
    formatted_w = map(x -> replace(x, 'j' => "im", '(' => "", ')' => ""), df.w)

    d = map(x -> parse(ComplexF64, x), formatted_d)
    w = map(x -> parse(ComplexF64, x), formatted_w)


    data = DataFrame(similarity=df.similarity)
    mapper = DataFrameMapper([
                            #[:w], StandardScaler()),
                            #([:frequency], StandardScaler())
                            ([:similarity], StandardScaler())
                            ]);
    mapper = fit_transform!(mapper, copy(data))
end

# Similary distance resonances (cos d_{jk})
function resonanceSimilarity(df)

    formatted_d = map(x -> replace(x, 'j' => "im", '(' => "", ')' => ""), df.d)
    formatted_w = map(x -> replace(x, 'j' => "im", '(' => "", ')' => ""), df.w)

    d = map(x -> parse(ComplexF64, x), formatted_d)
    w = map(x -> parse(ComplexF64, x), formatted_w)

    djdk = map((x,y) -> x.*y, d[1:end-1], d[2:end])
    diff_wjwk = diff(w)


    dj_absPow = abs.(d[1:end-1]).^2
    dk_absPow = abs.(d[2:end]).^2

    gj = df.decay[1:end-1]
    gk = df.decay[2:end]

    numerator = real(djdk./diff_wjwk)

    similarity = numerator ./ (dj_absPow./gj).*(dk_absPow./gk)

    # last element has 0 similarity with first one
    push!(similarity,0)

    similarity[similarity.>=1] .= 1
    similarity[similarity.<=-1] .= -1
    #similarity[.>]

    maximum(similarity) = 0

    #similarity = map(x -> if (x >= 1) x = 0 end, similarity)

    similarity
end

##########################################################################################################


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


############################## PLOT FUNCTIONS ###############################

function plotharmonic(df)
    # non-harmonic data
    trace1 = scatter(
    mode="markers",
    x=df[df.harmonic .== -1, :].onset,
    y=df[df.harmonic .== -1, :].frequency,
    opacity=0.5,
    marker=attr(
        size=2,
        color="646FFB"
    ),
    name="data"
    )

    # harmonic data
    trace2 = scatter(
    mode="markers",
    x=df[df.harmonic .!== -1, :].onset,
    y=df[df.harmonic .!== -1, :].frequency,
    opacity=1,
    marker=attr(
        color=df[df.harmonic .!== -1, :].harmonic,
        size=4
    ),
    name="overtones"
    )

    p = plot([trace1, trace2], Layout(title="Overtone seperation", yaxis_title="Frequency (Hz)", xaxis_title="Time"))
    savefig(p, "overtones.png")

    p
end

function plotf0(df)
    # https://plotly.com/julia/reference/scatter3d/
    p = plot(
        df, 
        Layout(scene = attr(
                        xaxis_title="Time (s)",
                        yaxis_title="Frequency (Hz)",
                        zaxis_title="Power"
                        ),
                        #margin=attr(r=100, b=150, l=50, t=50)
                        ),
        x=:onset, 
        y=:frequency, 
        z=:power, 
        color=:f0, # choose one of the two, dependent on harmonic
        type="scatter3d", 
        mode="markers", 
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

    # path = "./code/add-ons/plots-demo/"*string(EPS)*"-"*string(PTS)*".png"
    # savefig(p, path)

    # open("./example.html", "w") do io
    #     PlotlyBase.to_html(io, p.plot)
    # end

    p
end

#############################################################################


# accuracy must be a value between 10 and 50, since 0.1 <= eps <= 0.5
# The higher the value, the less accuracy (just inverse for user later), 
# mainly used for pieces where notes vary strongly in time
# note: increase accuracy increases running time as well
main(PATH, 6) # MOET NOG LAGER VOOR DIE SYNTHETISCHE NUMMERS
