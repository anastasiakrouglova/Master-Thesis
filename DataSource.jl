module DataSource

using Chakra
using DataFrames
using CSV

push!(LOAD_PATH, ".")
using Resonances
#include("./Resonances.jl")

# make main map the current directory
dir = @__DIR__

filename = "flute-a4.csv" # data is too big
#filename = "filtered-clustered-flute_a4.csv"
filepath = joinpath(dir,"fpt/data/output/",filename)

__data__ = Resonances.DRSHierarchy(filepath)

end # module
