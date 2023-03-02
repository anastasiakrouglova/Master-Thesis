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

struct SliceId <: Id
    value::Int
end

struct SliceSequenceId <: Id
    value::Int
end

struct PairId <: Id
    value::Int
end

resId(i::Int) = ResonanceId(i)
sliceId(i::Int) = SliceId(i)
sliceSeqId(i::Int) = SliceSequenceId(i)
pairId(i::Int) = PairId(i)
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
        df[!,:id] = collect(1:size(df)[1]) # add res id

        sliceIds = [floor(Int, (o / d)) for (o, d) in zip(df.onset, df.duration)]
        df[!, :sliceId] = sliceIds

        pairIds =  [ceil(Int, (id / 2)) for (id) in df.id]
        df[!, :pairId] = pairIds

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
        #values = [id.value for id in ids]

        df = filter(:id => o -> o in values, dataset.data)
        return new(ids, df)
    end
end

# a slice in the spectrogram (based on time)
struct Slice <: ResonanceSet
    sliceId::SliceId
    resonances::DataFrame
    Slice(sliceId::SliceId,dataset::DataSet) = begin
        #duration = dataset.data.duration[1]
        df = filter(:sliceId => o -> o == sliceId.value, dataset.data)
        return new(sliceId,df)
    end
end


struct Pair <: ResonanceSet
    pairId::PairId
    resonances::DataFrame

    Pair(pairId::PairId, dataset::DataSet) = begin
        df = filter(:pairId => p -> p == pairId.value, dataset.data)
        return new(pairId,df)
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
            # TODO: Slices enumeraten 
            # SliceSequence
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
    min::Union{Real,Colon}
    max::Union{Real,Colon}
    resonances::DataFrame
    FrequencyBand(min::Union{Real,Colon}, max::Union{Real,Colon}, dataset::DataSet) = begin
        if (typeof(max) == Colon)
            df = filter(:frequency => f -> min <= f, dataset.data)
        elseif (typeof(min) == Colon)
            df = filter(:frequency => f -> f < max, dataset.data)
        else
            df = filter(:frequency => f -> min <= f <= max, dataset.data)
        end

        return new(min, max, df)
    end


    FrequencyBand(s::String, dataset::DataSet) = begin
        if (s == "pos")
            FrequencyBand(0, :, dataset::DataSet)
        elseif (s == "neg")
            FrequencyBand(:, 0, dataset::DataSet)
        elseif (s == "null")
            FrequencyBand(0, 0, dataset::DataSet)
        else
            error("please put in pos/neg/null")
        end
    end
end



########################################################################################################
########################################################################################################
###################################   OPERATIONS  ######################################################
########################################################################################################
########################################################################################################

####################################  find a resonance by id ###########################################
## Application
### i = Flute.Resonances.id(<id1>) 
### Resonances.findResonancesbyId(i, <Module (e.g. Flute)>)

Chakra.fnd(x::ResonanceId, m::DataSet) = begin
    i = findall(==(x.value),m.data.id)
    isempty(i) ? none : Resonance(m.data[i[1],:])
end

# Chakra.fnd(x::PairId, m::DataSet) = begin
#     i = findall(==(x.value),m.data.id)
#     isempty(i) ? none : Pair(m.data[i[1],:])
# end

# Operations on data
#Chakra.fnd(x::ResonanceId, m::ResonanceHierarchy) = Chakra.fnd(x,m.dataset)


# function findResonanceById(x::ResonanceId, m::Module)
#     i = findall(==(x.value),m.__data__.data.id)
#     return isempty(i) ? none : Resonance(m.__data__.data[i[1],:])
# end



#################################  find a resonance by multiple ids ######################################
function findResonancesByIds(ids::Vector{ResonanceId}, m::Module)
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


# function findSliceById(x::SliceId, m::Module)
#     i = findall(==(x.value),m.__data__.data.onset[])
#     return isempty(i) ? none : Resonance(m.__data__.data[i[1],:])
# end


#############################  filter on frequency (Function overloading) ###############################
function signFrequencies(s::String, m::Module) 
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


function frequencyBand(min::Union{Real,Colon}, max::Union{Real,Colon}, m::Module)
    """
    Find frequencies between a minimum and maximum frequency value.

    # Arguments
    `min::Union{Real,Colon}, max::Union{Real,Colon}` = contains in the first element the minimum and in the second the maximum value for constraining the frequency band

    # Examples
    ```
    julia> Resonances.frequencyBand([3, 10000], Flute)
    ```
    """
    return FrequencyBand(min, max, m.__data__) 
end

function frequencyBand(s::String, m::Module)
    return FrequencyBand(s, m.__data__) 
end



################################### filter on slice ######################################################
function getSlice(x::SliceId, m::Module) 
    # TODO: return correct slice, not only equal to first one!
    return Slice(x, m.__data__)
end

function getSlice(min::SliceId, max::SliceId, m::Module)
    return SliceSequence(min, max, m.__data__)
end

function getSlice(min::SliceId, max::Colon, m::Module)
    return SliceSequence(min, max, m.__data__)
end

function getSlice(min::Colon, max::SliceId, m::Module)
    return SliceSequence(min, max, m.__data__)
end

# function getSliceSequence(min::SliceId, max::SliceId, ::Module)
#     return SliceSequence(min, max, m.__data__)
# end



Chakra.pts(x::Resonance)::Vector{Id} = Id[] # empty because the resonance is the smallest constituent
Chakra.pts( x::Resonances.ResonanceCollection) = x.ids
Chakra.pts(x::Slice)::Vector{Id} = pairId.(unique(x.resonances.pairId)) 
# TODO: Don't understand
Chakra.pts(x::SliceSequence)::Vector{Id} = sliceId.(unique(x.resonances.sliceId))  

Chakra.pts(x::Frequencies)::Vector{Id} = resId.(x.resonances.id) 
Chakra.pts(x::FrequencyBand)::Vector{Id} = resId.(x.resonances.id)  

Chakra.pts(x::Pair)::Vector{Id} = resId.(x.resonances.id)

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