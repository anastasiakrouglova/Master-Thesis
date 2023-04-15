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
filenameH = "filtered-clustered-flute_a4.csv"

filepath = joinpath(dir,"fpt/data/output/",filename)
filepathHarmonics = joinpath(dir,"fpt/data/output/",filenameH)

__data__ = Resonances.DRSHierarchy(filepath)
__harmonics__ = Resonances.DynRHierarchy(filepathHarmonics)

end # module
