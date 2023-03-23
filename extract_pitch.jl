using PlotlyJS, ClusterAnalysis, StatsBase, DataFrames, CSV

#raw = DataFrame(CSV.File("./data-ml/out.csv"))
raw = DataFrame(CSV.File("./fpt/data/output/flute-a4.csv"))

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

# Parameters of DBSCAN
ϵ = 0.5;
min_pts = 5;

# Denoise raw data
df = remove_noise(raw, 0.001, 2000)

# Convert data to a matrix
X = convert(Matrix, df[:,[1, 6, 8]])

# normalize matrix
dt = fit(ZScoreTransform, X, dims=1)
mat = StatsBase.transform(dt, X)

# Run DBSCAN 
m = dbscan(mat, ϵ, min_pts); #returns object dbscan!

# Put labels from clustering back to a dataframe
print(m.labels)
df[!,:cluster] = m.labels

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
