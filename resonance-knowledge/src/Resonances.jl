module Resonances

using DataFrames
using CSV

using Chakra

# Definiton abstract types
abstract type Id <: Chakra.Id end
abstract type Constituent <: Chakra.Constituent end
abstract type Hierarchy <: Chakra.Hierarchy end

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

struct SliceSequenceFirst <: ResonanceSet
    start::Int
    resonances::DataFrame
    SliceSequenceFirst(start::Int, dataset::DataSet) = begin
        df = filter(:onset => o -> start <= o, dataset.data)
        return new(start, df)
    end
end

struct SliceSequenceScnd <: ResonanceSet
    finish::Int
    resonances::DataFrame
    SliceSequenceScnd(finish::Int, dataset::DataSet) = begin
        df = filter(:onset => o -> o <= finish, dataset.data)
        return new(finish, df)
    end
end





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

# You will probably also need a type SliceId (or ResonanceSetId) --- NO

# Other types of resonance set? 
## Negative frequencies?  -- DONE
## Positive frequencies? -- DONE
## Sequences of slices?
## frequency bands?
# What about collections of resonance sets etc? 

# What about Hierarchies which contain resonance sets and other collectsion?
# For example:


########################################################################################################

Chakra.fnd(x::ResonanceId, m::DataSet) = begin
    i = findall(==(x.value),m.data.id)
    isempty(i) ? none : Resonance(m.data[i[1],:])
end

# Operations on data
Chakra.fnd(x::ResonanceId, m::ResonanceHierarchy) = Chakra.fnd(x,m.dataset)


# Todo: op bepaalde types filteren: bv onset, frequency
#Chakra.filter(x::Int, m::Module, t::String) = none




# filter on frequency: bool 1 = pos, 0 = neg
Chakra.filter(s::String, m::Module) = begin
    return Frequencies(s,m.__data__) #PosFrequencies(m.__data__)
end


# filter on a certain slice
Chakra.filter(x::Int, m::Module) = begin 
    # slice number is multiplied by the duration (how onset works)
    #i = findall((m.__data__.data.duration) == m.__data__.data.duration[1])
    #println(i)
    duration = m.__data__.data.duration[1]
    return Slice(x*duration, m.__data__)
    # TODO: return correct slice, not only equal to first one!
    # WEIRD: Gives also 185, which is not equal to first duration, but still works...
end




# Function overloading
function filterSlice(x::Int, m::Module) 
    return Slice(x*m.__data__.data.duration[1], m.__data__)
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


# Chakra.fnd(x::SliceId, m::ResonanceHierarchy) = Chakra.fnd(x,m.dataset)
# Chakra.fnd(x::Id,m::ResonanceHierarchy) = Base.get(m.structure,x,none)

Chakra.pts(x::Resonance)::Vector{Id} = Id[] # empty because the resonance is the smallest constituent
Chakra.pts(x::Slice)::Vector{Id} = id.(x.resonances.id)  

end




# # Operations on data
# Base.get(d::DataFrame,x::Int) = x > size(d)[1] ? none : d[x,:]
# Chakra.fnd(x::ResonanceId, m::DataSet) = (r = Base.get(m.data, x.index); r == none ? none : Resonance(r))

# # define a chakra-particle 
# Chakra.pts(x::Resonance)::Vector{<:Id} = Id[] # empty because doesn’t have any smaller atoms (onder in tree)


