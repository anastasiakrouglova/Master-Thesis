module Flute

using Chakra
using DataFrames
using CSV

include("./Resonances.jl")

dir = @__DIR__

filename = "flute_a4.csv"
filepath = joinpath(dir,"..",filename)

__data__ = Resonances.DataSet(filepath)

end # module
