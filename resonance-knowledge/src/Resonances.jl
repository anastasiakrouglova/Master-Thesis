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
    value::Int
end

resId(i::Int) = ResonanceId(i)
sliceId(i::Int) = SliceId(i)

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
    onset::SliceId
    resonances::DataFrame
    Slice(onset::SliceId,dataset::DataSet) = begin
        duration = dataset.data.duration[1]
        df = filter(:onset => o -> o == onset.value*duration, dataset.data)
        return new(onset,df)
    end
end

struct SliceSequence <: ResonanceSet
    start::Union{SliceId,Colon}
    finish::Union{SliceId,Colon}
    resonances::DataFrame
    SliceSequence(start::Union{SliceId,Colon}, finish::Union{SliceId,Colon}, dataset::DataSet) = begin
        # TODO: duration dependent on number, not just first element
        # slice number is multiplied by the duration (how onset works)
        duration = dataset.data.duration[1]

        if (typeof(start) == SliceId && typeof(finish) == SliceId)
            df = filter(:onset => o -> start.value*duration <= o <= finish.value*duration, dataset.data)
        elseif (typeof(start) == SliceId && typeof(finish) == Colon)
            df = filter(:onset => o -> start.value*duration <= o, dataset.data)
        elseif (typeof(start) == Colon && typeof(finish) == SliceId)
            df = filter(:onset => o -> o <= finish.value*duration, dataset.data)
        end
        return new(start,finish, df)
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
    min::Union{Int,Colon}
    max::Union{Int,Colon}
    resonances::DataFrame
    FrequencyBand(min::Int, max::Int, dataset::DataSet) = begin
            df = filter(:frequency => f -> min <= f <= max, dataset.data)
        return new(min, max, df)
    end
end



########################################################################################################
########################################################################################################
###################################   OPERATIONS  ######################################################
########################################################################################################
########################################################################################################

####################################  find a resonance by id ###########################################
## Application
#### i = Flute.Resonances.id(<id1>) 
#### Resonances.findResonancesbyId(i, <Module (e.g. Flute)>)
Chakra.fnd(x::ResonanceId, m::DataSet) = begin
    i = findall(==(x.value),m.data.id)
    isempty(i) ? none : Resonance(m.data[i[1],:])
end

# Operations on data
Chakra.fnd(x::ResonanceId, m::ResonanceHierarchy) = Chakra.fnd(x,m.dataset)


#################################  find a resonance by multiple ids ######################################
function findResonancesbyId(ids::Vector{ResonanceId}, m::Module)
    """
    Find resonances of multiple ids.

    # Examples
    ```jldoctest
    julia> i1 = Flute.Resonances.id(10) 
    julia> i2 = Flute.Resonances.id(20)
    julia> Resonances.findResonancesbyId([i1, i2], Flute)
    2×11 DataFrame:
    Row │ onset  duration  sample_rate  amplitude   phase      frequency  decay   ⋯
    │ Int64  Int64     Int64        Float64     Float64    Float64    Float64 ⋯
    ─────┼──────────────────────────────────────────────────────────────────────────
    1 │     0       512        44100  8.45435e-5  -0.766382   -20398.8  -137.77 ⋯
    2 │     0       512        44100  2.70559e-5   1.26059    -17333.4  -171.53
    ```
    """    
    ResonanceCollection(ids, m.__data__)
end

#############################  filter on frequency (Function overloading) ###############################
function filterFrequency(s::String, m::Module) 
    """
    Find frequencies with a certain characteristic

    # Arguments
    `s::String` = "pos" / "neg" / "null" # to filter on positive/negative/null frequencies

    # Examples
    ```
    julia> Flute.Resonances.id("pos", <Module>) 
    ```
    """    
    return Frequencies(s,m.__data__) 
end


function frequencyBand(band::Vector{Union{Int, Int}}, m::Module)
    """
    Find frequencies between a minimum and maximum frequency value.

    # Arguments
    `band::::Vector{Union{Int, Int}}` = contains in the first element the minimum and in the second the maximum value for constraining the frequency band

    # Examples
    ```
    julia> Resonances.frequencyBand([3, 10000], Flute)
    ```
    """
    return FrequencyBand(band[1], band[2], m.__data__) 
end



################################### filter on slice ######################################################
function getSlice(x::SliceId, m::Module) 
    # TODO: return correct slice, not only equal to first one!
    #duration = m.__data__.data.duration[1]
    return Slice(x, m.__data__)
end

function getSlice(seq::Vector{Any}, m::Module) 
    return SliceSequence(seq[1], seq[2], m.__data__)
end

function getSlice(seq::Vector{SliceId}, m::Module) 
    return SliceSequence(seq[1], seq[2], m.__data__)
end

function getSlice(seq::Vector{Colon}, m::Module) 
    return error("Please enter a vector of length 2 with a slideId and maximum 1 Colon.")
end


Chakra.pts(x::Resonance)::Vector{Id} = Id[] # empty because the resonance is the smallest constituent
Chakra.pts(x::Slice)::Vector{Id} = id.(x.resonances.id)  
#Chakra.pts(x::SliceSequence)::Vector{Id} = id.(x.resonances.id) # is this correct?


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
# What about collections of resonance sets etc? (dus rode bolletjes hoger)

# What about Hierarchies which contain resonance sets and other collectsion?
# For example: