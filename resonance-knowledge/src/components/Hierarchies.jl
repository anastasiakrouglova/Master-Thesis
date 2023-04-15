# Definition Hierarchies
struct DRSHierarchy <: Hierarchy
    data::DataFrame
    DRSHierarchy(filepath) = begin
        df = DataFrame(CSV.File(filepath))
        # Create a unique id for every dataframe
        df[!,:id] = collect(1:size(df)[1]) # add res id
        # add a sliceId column
        sliceIds = [floor(Int, (o / d)) for (o, d) in zip(df.onset, df.duration)]
        df[!, :sliceId] = sliceIds

        # add a pairId column only if complex-valued dataset


        if (df.amplitude[1] == df.amplitude[2])
            pairIds =  [ceil(Int, (id / 2)) for (id) in df.id]
            df[!, :pairId] = pairIds
        else
            print("Note: Your dataset does not contain complex values, so you won't need the Pair in the Hierarchy.")
        end
        
        new(df)
    end
end

# struct ResonanceHierarchy <: Hierarchy
#     dataset::DRSHierarchy #the atomic resonanaces
#     structure::Dict{Id,Constituent} # other constituents like slices etc
# end


# Hierarchy of dynamic resonances (other dimension than DRS)
struct DynRHierarchy <: Hierarchy
    data::DataFrame
    DynRHierarchy(filepath) = begin
        df = DataFrame(CSV.File(filepath))
        # A unique id is already created in the cluster algorithm to match the filtered data
        new(df)
    end
end

# struct DynResonanceSet <: Hierarchy
#     dataset::DynRHierarchy #the atomic resonanaces
#     structure::Dict{Id,Constituent} # other constituents like slices etc
# end