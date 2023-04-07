

#TODO: use other visualisation tool

# https://towardsdatascience.com/large-graph-visualization-tools-and-approaches-2b8758a1cd59


using Pkg; Pkg.instantiate()
using Chakra #, Charm
push!(LOAD_PATH, "./resonance-knowledge/src/") # TODO: load preprocessed data
using Resonances
push!(LOAD_PATH, ".") # TODO: load preprocessed data
using DataSource
# include("./clustering.jl")
using PyCall # call python functions in julia
using Conda

# Define drs
drs = DataSource.Resonances.drsId(1)
sliceIds = pts(drs, DataSource) # vector of slice Id's


# 1. Plot 1 bol voor drs object 
# 2. Add label to node
# 3. plot bollen voor sliceIDs (+labels)
# 2. Add edges

net = pyimport("pyvis")
# nx = pyimport("networkx")
g = net.network.Network(notebook=true, cdn_resources="remote")


# convert objects to strings

g.add_nodes(["DRS"])
g.add_nodes(["sliceID(1)", "sliceID(2)"])

g.add_edge("DRS", "sliceID(1)")
g.add_edge("DRS", "sliceID(2)")
g.get_node("sliceID(2)")

# save html
g.show("aaaa.html") 






