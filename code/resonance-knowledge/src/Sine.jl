module Sine

using Chakra, DataFrames, CSV
push!(LOAD_PATH, ".")
using Resonances
#include("./Resonances.jl")

dir = @__DIR__

filename = "data/sine440.csv"
filepath = joinpath(dir,"..",filename)

__data__ = Resonances.DRSHierarchy(filepath)

end # module