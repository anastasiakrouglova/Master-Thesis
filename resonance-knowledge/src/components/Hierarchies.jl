# Definition Hierarchies
struct DataSet <: Hierarchy
    data::DataFrame
    DataSet(filepath) = begin
        df = DataFrame(CSV.File(filepath))
        # Create a unique id for every dataframe
        df[!,:id] = collect(1:size(df)[1]) # add res id
        # add a sliceId column
        sliceIds = [floor(Int, (o / d)) for (o, d) in zip(df.onset, df.duration)]
        df[!, :sliceId] = sliceIds
        # add a pairId column
        pairIds =  [ceil(Int, (id / 2)) for (id) in df.id]
        df[!, :pairId] = pairIds

        new(df)
    end
end

struct ResonanceHierarchy <: Hierarchy
    dataset::DataSet #the atomic resonanaces
    structure::Dict{Id,Constituent} # other constituents like slices etc
end
