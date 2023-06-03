module Flute

using Chakra
using DataFrames
using CSV

push!(LOAD_PATH, ".")
using Resonances
#include("./Resonances.jl")

dir = @__DIR__

filename = "data/flute_syrinx_1.csv"
filepath = joinpath(dir,"..",filename)

__data__ = Resonances.DRSHierarchy(filepath)
__fundamental__ = Resonances.FUNDHierarchy(filepath)
__harmonics__ = Resonances.HARMHierarchy(filepath)
__real__ = Resonances.REALHierarchy(filepath)

end # module
