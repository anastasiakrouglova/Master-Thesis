

# This file creates 2 CSV files for a graph visualisation of the resonance-knowledge hierarchy.
# Import both files in Gephi (https://gephi.org/) and download the following plugin: (https://gephi.org/plugins//#/plugin/network-splitter-3d)
# Rename the "Level" column in the node.CSV to Level[Z] and run first "ForseAtlas 2" in the layout section, before applying the "Network Splitter 3D" layout.

# For more information: https://gephi.org/users/tutorial-layouts/ and 
# For examples of the visualisation: Masters Thesis of Anastasia Krouglova (github link TODO)

using Pkg; Pkg.instantiate()
using Chakra #, Charm
push!(LOAD_PATH, "./resonance-knowledge/src/") # TODO: load preprocessed data
using Resonances
push!(LOAD_PATH, ".") # TODO: load preprocessed data
using DataSource
# include("./clustering.jl")
using PyCall # call python functions in julia
using Conda

using DataFrames, CSV


function idToVectorString(ids)
    """
    Input: id or Vector{ids} 
    Return: Vector{String}
    """
    if (typeof(ids) == Vector{Resonances.Id})
        ids_vec = map(id -> 
        begin
            t = repr("text/plain", id)
            r = r"Resonances."
            id = replace(t,r => "")
            
        end, ids)

        ids_vec
    else
        t = repr("text/plain", drs)
        r = r"Resonances."
        s = replace(t,r => "")

        id_vec = Vector{String}()
        push!(id_vec,s)
    end
end


# Define parts of the Hierarchy
drs = DataSource.Resonances.drsId(1)
sliceIds = pts(drs, DataSource) # vector of slice Id's

drsIds_labels = idToVectorString(drs)
slideIds_labels = idToVectorString(sliceIds)
pairIds_labels = Vector{Vector}()
resonanceIds_labels = Vector{Vector}()

l = length(sliceIds)
for i in 1:l
    idsi = pts(sliceIds[i], DataSource) # idsI can be whether a pair or a resonance

    if (typeof(idsi[1]) == Resonances.PairId)
        # If pair is needed in the hierarchy (complex valued)
        pairIdsi_labels = idToVectorString(idsi)
        push!(pairIds_labels, pairIdsi_labels)

        lp = length(idsi)
        for j in 1:lp
            resIdsi = pts(idsi[j], DataSource)
            resIdsi_labels = idToVectorString(resIdsi)
            push!(resonanceIds_labels, resIdsi_labels)
        end
    else
        # If only real part
        resIdsi_labels = idToVectorString(idsi)
        push!(resonanceIds_labels, resIdsi_labels)
    end
end



function nodeCSV()
    df_node = DataFrame(Id=drsIds_labels[1], Level=4)

    l = length(slideIds_labels)
    for i in 1:l
        push!(df_node, (slideIds_labels[i], 3))

        if(isempty(pairIds_labels))
            # If not complex valued hierarchy
            for j in 1:length(resonanceIds_labels[i][:])
                push!(df_node, (resonanceIds_labels[i][j], 2))
            end
        else
            # If complex valued hierarchy
            for j in 1:length(pairIds_labels[i][:])
            push!(df_node, (pairIds_labels[i][j], 2))
                for k in 1:length(resonanceIds_labels[j][:])
                    push!(df_node, (resonanceIds_labels[j][k], 1))
                end
            end
        end
    end

    CSV.write("./nodes2.csv", df_node)
end

function edgeCSV()
    df_edge = DataFrame(Source=drsIds_labels[1], Target=slideIds_labels)

    l = length(slideIds_labels)
    for i in 1:l
        if(isempty(pairIds_labels))
            # if not complex valued hierarchy
            for j in 1:length(resonanceIds_labels[i][:])
                push!(df_edge, (slideIds_labels[i], resonanceIds_labels[i][j]))
            end
        else
            # if not complex valued hierarchy
            for j in 1:length(pairIds_labels[i][:])
                push!(df_edge, (slideIds_labels[i], pairIds_labels[i][j]))

                for k in 1:length(resonanceIds_labels[j][:])
                    push!(df_edge, (pairIds_labels[i][j], resonanceIds_labels[j][k]))
                end
            end
        end

    end

    CSV.write("./edges2.csv", df_edge)
end

nodeCSV()
edgeCSV()

