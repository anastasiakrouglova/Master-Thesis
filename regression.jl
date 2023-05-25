# using PlotlyJS, ClusterAnalysis, StatsBase, DataFrames, CSV, LinearAlgebra
# using PyCall
# using Conda
# using ScikitLearn

# filename = "flute_syrinx_1_f0"
# PATH = "./fpt/data/output/scores/clustered/" * filename * ".csv"


# Create a sample dataset
data = [0, -0.0, 5, 2, -0.5, 0]

for value in data
    if iszero(value)
        println("$value is -0")
    elseif value == 0
        println("$value is 0")
    else
        println("$value is not 0 or -0")
    end
end