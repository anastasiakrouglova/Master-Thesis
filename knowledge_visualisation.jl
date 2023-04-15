

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
        t = repr("text/plain", ids)
        r = r"Resonances."
        s = replace(t,r => "")

        id_vec = Vector{String}()
        push!(id_vec,s)
    end
end

function constituentsToLabels(origin, output) 
    lh = length(origin)
    for i in 1:lh
        idsi = pts(origin[i], DataSource)

        if (typeof(idsi[1]) == Resonances.PairId)
            # for the DRS (complex valued) hierarchy containing pairs
            pairIdsi_labels = idToVectorString(idsi)
            push!(pairIds_labels, pairIdsi_labels)
    
            lp = length(idsi)
            for j in 1:lp
                resIdsi = pts(idsi[j], DataSource)
                resIdsi_labels = idToVectorString(resIdsi)
                push!(resonanceIds_labels, resIdsi_labels)
            end
        else
    
        resIdsi_labels = idToVectorString(idsi)
        push!(output, resIdsi_labels)

        end
    end
end


# Define parts of the DRS Hierarchy
drs = DataSource.Resonances.drsId(1)
sliceIds = pts(drs, DataSource) # vector of slice Id's

# Define parts of the DynR Hierarchy
dynr = DataSource.Resonances.dynRId(1)
harmonicIds = pts(dynr, DataSource)

# Convert Hierarchical components to strings for visualisation
dynrIds_labels = idToVectorString(dynr)
harmonicIds_labels = idToVectorString(harmonicIds)
dynamicResonanceIds_labels = Vector{Vector}()

drsIds_labels = idToVectorString(drs)
sliceIds_labels = idToVectorString(sliceIds)
pairIds_labels = Vector{Vector}()
resonanceIds_labels = Vector{Vector}()

# convert DRS and DynR resonances & pairs to an array of strings
constituentsToLabels(sliceIds, resonanceIds_labels) 
constituentsToLabels(harmonicIds, dynamicResonanceIds_labels) 


function nodeCSV()
    df_node = DataFrame(Id=drsIds_labels[1], Level=4)

    # Add DynR node
    push!(df_node, (dynrIds_labels[1], 4))

    # Add Harmonic nodes
    ld = length(harmonicIds_labels) - 1 # -1 due to noisebatch 
    for i in 1:ld
        push!(df_node, (harmonicIds_labels[i], 3)) 
    end

    resStart = 0
    l = length(sliceIds_labels) # 380
    for i in 1:l
        push!(df_node, (sliceIds_labels[i], 3))

        # if(isempty(pairIds_labels))
        #     # If not complex valued hierarchy: Dynr
        #     for j in 1:length(resonanceIds_labels[i][:]) 
        #         push!(df_node, (resonanceIds_labels[i][j], 2))
        #     end
        # else
            # If complex valued hierarchy: DRS

        for j in 1:length(pairIds_labels[i][:])
            push!(df_node, (pairIds_labels[i][j], 2)) # 44, 41, 47, 55, ...

            lk = length(resonanceIds_labels[j+resStart][:])
            for k in 1:lk
                push!(df_node, (resonanceIds_labels[j+resStart][k], 1))
            end
        end
        resStart = resStart + length(pairIds_labels[i])
        # end
    end

    CSV.write("./gephi/nodesDynR.csv", df_node)
end

function edgeCSV()
    df_edge = DataFrame(Source=drsIds_labels[1], Target=sliceIds_labels)
    resStart = 0

    # Push edges of DynR
    lh = length(harmonicIds_labels) -1
    for i in 1:lh
        for j in 1:length(dynamicResonanceIds_labels[i][:])
            push!(df_edge, (harmonicIds_labels[i], dynamicResonanceIds_labels[i][j]))
        end
    end

    # Push edges of DRS
    l = length(sliceIds_labels)
    for i in 1:l
        # if(isempty(pairIds_labels))
        #     # if not complex valued hierarchy
        #     for j in 1:length(resonanceIds_labels[i][:])
        #         push!(df_edge, (sliceIds_labels[i], resonanceIds_labels[i][j]))
        #     end
        # else
            # if not complex valued hierarchy
        for j in 1:length(pairIds_labels[i][:])
            push!(df_edge, (sliceIds_labels[i], pairIds_labels[i][j]))

            for k in 1:length(resonanceIds_labels[j+resStart][:])
                push!(df_edge, (pairIds_labels[i][j], resonanceIds_labels[j+resStart][k]))
            end
        end
        resStart = resStart + length(pairIds_labels[i])
        # end
    end

    CSV.write("./gephi/edgesDynR.csv", df_edge)
end


nodeCSV()
edgeCSV()

