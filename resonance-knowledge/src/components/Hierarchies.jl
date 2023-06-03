# Definition Hierarchies

# The discrete resonance spectrum Hierarchy represents the hierarchy of a spectrogram
struct DRSHierarchy <: Hierarchy
    data::DataFrame
    DRSHierarchy(filepath) = begin
        df = DataFrame(CSV.File(filepath))
        # Create a unique id for every dataframe
        # df[!,:id] = collect(1:size(df)[1]) # add res id
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



# Contains only the real part of the resonances
struct REALHierarchy <: Hierarchy
     data::DataFrame
     REALHierarchy(filepath) = begin
         raw = DataFrame(CSV.File(filepath))
         # Filter on only positive resonances and > 0
         df = filter(:frequency => x -> x > 0, raw)

         new(df)
     end
end


struct FUNDHierarchy <: Hierarchy
    data::DataFrame #the atomic resonanaces

    FUNDHierarchy(filepath) = begin
        df = DataFrame(CSV.File(filepath))
        df = filter(:frequency => x -> x > 0, df)
        df = filter(:f0 => x -> x !== 0, df)
        df = filter(:f0 => x -> x !== -1, df)

        df = DataFrame(id=df.id, f0=df.f0)
        new(df)
    end
end

# Returns the harmonic overtones of a fundamental 
struct HARMHierarchy <: Hierarchy
    data::DataFrame #the atomic resonanaces

    HARMHierarchy(filepath) = begin
        df = DataFrame(CSV.File(filepath))
        df = filter(:frequency => x -> x > 0, df)
        # df = filter(:f0 => x -> x == 0, df) # exclude the fundamental tone for the overtones
        df = filter(:harmonic => x -> x !== 0, df)
        df = filter(:harmonic => x -> x !== -1, df)

        df = DataFrame(id=df.id, harmonic=df.harmonic)
        new(df)
    end
end



# struct DynResonanceSet <: Hierarchy
#     dataset::DynRHierarchy #the atomic resonanaces
#     structure::Dict{Id,Constituent} # other constituents like slices etc
# end