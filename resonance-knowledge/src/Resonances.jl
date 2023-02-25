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

# You will probably also need a type SliceId (or ResonanceSetId)

# Other types of resonance set? 
## Negative frequencies? 
## Positive frequencies?
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



# filter on onset to get a certain slice
Chakra.filter(x::Int, m::Module) = begin 
    return Slice(x, m.__data__)
end



#Chakra.filter(x::SliceId, m::Tuple) = filter(x, m)

#Chakra.filter(x::SliceId, m::DataSet) = Chakra.filter(x,m.data.onset)

# Chakra.fnd(x::SliceId, m::DataSet) = begin
#     i = findall(==(x.onset),m.data.onset)
#     isempty(i) ? none : Slice(x, m)
# end

# Chakra.fnd(x::SliceId, m::ResonanceHierarchy) = Chakra.fnd(x,m.dataset)
# Chakra.fnd(x::Id,m::ResonanceHierarchy) = Base.get(m.structure,x,none)

Chakra.pts(x::Resonance)::Vector{Id} = Id[] # empty because the resonance is the smallest constituent
Chakra.pts(x::Slice)::Vector{Id} = id.(x.resonances.id)  

end




# # An abstract type for filtering of resonances
# abstract type ResonanceSet <: Constituent end

# struct Slice <: ResonanceSet
#     onset::Float64
#     data::DataFrame
# end

# struct PositiveResonances <: ResonanceSet
#     onset::Float64
#     data::DataFrame
# end

# struct NegativeResonances <: ResonanceSet
#     onset::Float64
#     data::DataFrame
# end


# # Operations on data
# Base.get(d::DataFrame,x::Int) = x > size(d)[1] ? none : d[x,:]
# Chakra.fnd(x::ResonanceId, m::DataSet) = (r = Base.get(m.data, x.index); r == none ? none : Resonance(r))

# # define a chakra-particle 
# Chakra.pts(x::Resonance)::Vector{<:Id} = Id[] # empty because doesn’t have any smaller atoms (onder in tree)


# end
