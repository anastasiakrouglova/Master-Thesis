module Resonances

using DataFrames
using CSV

using Chakra

# Definiton abstract types
abstract type Id <: Chakra.Id end
abstract type Constituent <: Chakra.Constituent end
abstract type Hierarchy <: Chakra.Hierarchy end


########################################################################################################
########################################################################################################
################################  ID's, constituents, hierearchies #####################################
########################################################################################################
########################################################################################################

# Definition Ids
struct ResonanceId <: Id
    value::Int
end

#TODO ResonancesId 
    # value::List ([1:4] bv)

struct SliceId <: Id
    onset::Int
end

id(i::Int) = ResonanceId(i)
sId(i::Int) = SliceId(i)

# Definition Constituents
struct Resonance <: Constituent
    data::DataFrameRow
end



# Definition Hierarchies
struct DataSet <: Hierarchy
    data::DataFrame
    DataSet(filepath) = begin
        df = DataFrame(CSV.File(filepath))
        # Create a unique id for every dataframe
        df[!,:id] = collect(1:size(df)[1])
        new(df)
    end
end

struct ResonanceHierarchy <: Hierarchy
    dataset::DataSet #the atomic resonanaces
    structure::Dict{Id,Constituent} # other constituents like slices etc
end


########################################################################################################


# An abstract type to filter resonances
abstract type ResonanceSet end


struct ResonanceCollection <: ResonanceSet
    ids::Vector{ResonanceId}
    data::DataFrame
    ResonanceCollection(ids::Vector{ResonanceId}, dataset::DataSet) = begin
        values = Vector{Int64}()
        for (index, val) in enumerate(ids)
            append!(values, ids[index].value)
        end

        df = filter(:id => o -> o in values, dataset.data)
        return new(ids, df)
    end
end

# a slice in the spectrogram (based on time)
struct Slice <: ResonanceSet
    onset::Int
    resonances::DataFrame
    Slice(onset::Int,dataset::DataSet) = begin
        df = filter(:onset => o -> o == onset, dataset.data)
        return new(onset,df)
    end
end

struct SliceSequence <: ResonanceSet
    start::Int
    finish::Int
    resonances::DataFrame
    SliceSequence(start::Int, finish::Int, dataset::DataSet) = begin
        df = filter(:onset => o -> start <= o <= finish, dataset.data)
        return new(start,finish, df)
    end
end

# for input such as [x, :]
struct SliceSequenceFirst <: ResonanceSet
    start::Int
    resonances::DataFrame
    SliceSequenceFirst(start::Int, dataset::DataSet) = begin
        df = filter(:onset => o -> start <= o, dataset.data)
        return new(start, df)
    end
end

# for input such as [:, x]
struct SliceSequenceScnd <: ResonanceSet
    finish::Int
    resonances::DataFrame
    SliceSequenceScnd(finish::Int, dataset::DataSet) = begin
        df = filter(:onset => o -> o <= finish, dataset.data)
        return new(finish, df)
    end
end


# filter on positive, negative and 0 frequencies
struct Frequencies <: ResonanceSet
    typeFreq::String
    resonances::DataFrame
    Frequencies(typeFreq::String, dataset::DataSet) = begin
        if (typeFreq == "pos")
            df = filter(:frequency => f -> f > 0, dataset.data)
        elseif (typeFreq == "neg")
            df = filter(:frequency => f -> f < 0, dataset.data)
        elseif (typeFreq == "null")
            df = filter(:frequency => f -> f == 0, dataset.data)
        else
            df = null
        end
        return new(typeFreq, df)
    end
end

# frequency between [x, y]
struct FrequencyBand <: ResonanceSet
    start::Int
    finish::Int
    resonances::DataFrame
    FrequencyBand(start::Int, finish::Int, dataset::DataSet) = begin
            df = filter(:frequency => f -> start <= f <= finish, dataset.data)
        return new(start, finish, df)
    end
end






########################################################################################################
########################################################################################################
###################################   OPERATIONS  ######################################################
########################################################################################################
########################################################################################################

####################################  find a resonance by id ###########################################
Chakra.fnd(x::ResonanceId, m::DataSet) = begin
    i = findall(==(x.value),m.data.id)
    isempty(i) ? none : Resonance(m.data[i[1],:])
end

# Chakra.fnd(seq::Vector{ResonanceId}, m::Module) = begin
#     for j in seq
#         i = findall(==(seq[j].value),m.__data__.data.id)
#         isempty(i) ? none : Resonance(m.__data__.data[i[1],:])
#     end
# end

# Operations on data
Chakra.fnd(x::ResonanceId, m::ResonanceHierarchy) = Chakra.fnd(x,m.dataset)
#Chakra.fnd(x::ResonanceId, m::ResonanceHierarchy) = Chakra.fnd(x,m.dataset)



# [2, 4 , 78, 3]
function findResonancesbyId(ids::Vector{ResonanceId}, m::Module)
    #for j in seq
    # i = findall(==(seq[1].value),m.__data__.data.id)
    # j = findall(==(seq[2].value),m.__data__.data.id)

    #isempty(i) ? none : 
    
    ResonanceCollection(ids, m.__data__)
        #println(i)
    #end

    #return 5
end




#############################  filter on frequency (Function overloading) ###############################
function filterFrequency(s::String, m::Module)
    return Frequencies(s,m.__data__) 
end


function filterFrequency(band::Vector{Union{Int, Int}}, m::Module)
    return FrequencyBand(band[1], band[2], m.__data__) 
end


################################### filter on slice ######################################################
function filterSlice(x::Int, m::Module) 
    # TODO: return correct slice, not only equal to first one!
    duration = m.__data__.data.duration[1]
    return Slice(x*duration, m.__data__)
end

function filterSlice(seq::Vector{Union{Int, Int}}, m::Module) 
    duration = m.__data__.data.duration[1] # slice number is multiplied by the duration (how onset works)
    return SliceSequence(seq[1]*duration, seq[2]*duration, m.__data__)
end

function filterSlice(seq::Vector{Any}, m::Module) 
    duration = m.__data__.data.duration[1]
    if (typeof(seq[2]) == Colon)
        return SliceSequenceFirst(seq[1]*duration, m.__data__)
    elseif (typeof(seq[1]) == Colon)
        return SliceSequenceScnd(seq[2]*duration, m.__data__)
    end
end


Chakra.pts(x::Resonance)::Vector{Id} = Id[] # empty because the resonance is the smallest constituent
Chakra.pts(x::Slice)::Vector{Id} = id.(x.resonances.id)  

end



########################################################################################################
########################################################################################################
###################################      TODO     ######################################################
########################################################################################################
########################################################################################################

# You will probably also need a type SliceId (or ResonanceSetId) --- NO

# Other types of resonance set? 
## Negative frequencies?  -- DONE
## Positive frequencies? -- DONE
## Sequences of slices? -- DONE
## frequency bands? -- DONE
# What about collections of resonance sets etc? 

# What about Hierarchies which contain resonance sets and other collectsion?
# For example: