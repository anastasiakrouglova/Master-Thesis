module Flute

using Chakra
using DataFrames
using CSV

push!(LOAD_PATH, ".")
using Resonances
#include("./Resonances.jl")

dir = @__DIR__

filename = "flute_a4.csv"
filepath = joinpath(dir,"..",filename)

__data__ = Resonances.DataSet(filepath)

end # module
