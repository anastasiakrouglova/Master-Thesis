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
    min::Int
    max::Int
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
## Application
#### Flute.Resonances.id("pos", <Module>) 
function filterFrequency(s::String, m::Module) # s can be "pos"/"neg"/"null"
    """
    Find frequencies with a certain characteristic

    # Variables
    s = "pos" / "neg" / "null" # to filter on positive/negative/null frequencies

    # Examples
    ```
    ```
    """    
    return Frequencies(s,m.__data__) 
end


function filterFrequency(band::Vector{Union{Int, Int}}, m::Module)
    """
    Find frequencies between a minimum and maximum frequency.

    # Variables
    band = a minimum and maximum value integer

    # Examples
    ```
    ```
    """
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