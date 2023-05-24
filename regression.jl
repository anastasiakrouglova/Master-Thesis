using PlotlyJS, ClusterAnalysis, StatsBase, DataFrames, CSV, LinearAlgebra
using PyCall
using Conda
using ScikitLearn

filename = "flute_syrinx_1_f0"
PATH = "./fpt/data/output/scores/clustered/" * filename * ".csv"


