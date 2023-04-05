using Chakra #, Charm
push!(LOAD_PATH, "./resonance-knowledge/src/") # TODO: load preprocessed data
using Resonances
push!(LOAD_PATH, ".") # TODO: load preprocessed data
using DataSource
# include("./clustering.jl")

# Define drs
drs = DataSource.Resonances.drsId(1)

sliceIds = pts(drs, DataSource) # vector of slice Id's



# 1. Plot 1 bol voor drs object 
# 2. Add label to node
# 3. plot bollen voor sliceIDs (+labels)
# 2. Add edges


# load makie libraries
using CairoMakie
using GraphMakie
using Graphs

g = wheel_graph(10)
f, ax, p = graphplot(g)
hidedecorations!(ax); hidespines!(ax)
ax.aspect = DataAspect()

graphplot(g)






