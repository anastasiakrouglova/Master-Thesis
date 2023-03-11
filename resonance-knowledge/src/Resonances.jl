module Resonances

using DataFrames
using CSV
using Chakra

# Definiton abstract types
abstract type Id <: Chakra.Id end
abstract type Constituent <: Chakra.Constituent end
abstract type Hierarchy <: Chakra.Hierarchy end


include("components/Ids.jl")
include("components/Hierarchies.jl")
include("components/Constituents.jl")


########################################################################################################
########################################################################################################
###################################   OPERATIONS  ######################################################
########################################################################################################
########################################################################################################

####################################  find a resonance by id ###########################################
Chakra.fnd(x::ResonanceId, m::DRSHierarchy) = begin
    i = findall(==(x.value),m.data.id)
    isempty(i) ? none : Resonance(m.data[i[1],:])
end

Chakra.fnd(x::PairId, m::DRSHierarchy) = begin
    Pair(x, m)
end

Chakra.fnd(x::SliceId, m::DRSHierarchy) = begin
    Slice(x, m)
end

Chakra.fnd(x::DRSId, m::DRSHierarchy) = begin
    DRS(x, m)
end


# Chakra.fnd(x::SliceSequenceId, m::DataSet) = begin
#     #Chakra.pts(x::SliceSequence)::Vector{Id} = sliceId.(unique(x.resonances.sliceId))  
#     getSlice(min::SliceId, max::Colon, m::Module)
# end

# Operations on data
#Chakra.fnd(x::ResonanceId, m::ResonanceHierarchy) = Chakra.fnd(x,m.dataset)




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



#############################  filter on frequency (Function overloading) ###############################
function getFrequencyBand(min::Union{Real,Colon}, max::Union{Real,Colon}, m::Module)
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

function getFrequencyBand(s::String, m::Module)
    return FrequencyBand(s, m.__data__) 
end



################################### filter on slice ######################################################
function getSlice(x::SliceId, m::Module) 
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



Chakra.pts(x::Resonance)::Vector{Id} = Id[] # empty because the resonance is the smallest constituent
Chakra.pts( x::Resonances.ResonanceCollection) = x.ids

Chakra.pts(x::Slice)::Vector{Id} = pairId.(unique(x.resonances.pairId)) 
Chakra.pts(x::SliceSequence)::Vector{Id} = sliceId.(unique(x.resonances.sliceId))  
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