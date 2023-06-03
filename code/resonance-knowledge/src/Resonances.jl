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


# One dimensional DRS structure in time
Chakra.pts(x::Resonance)::Vector{Id} = Id[]  # empty because the resonance is the smallest constituent
Chakra.pts(x::Pair)::Vector{Id} = resonanceId.(x.resonances.id)

Chakra.pts(x::Slice)::Vector{Id} = begin
    if("pairId" in names(x.resonances)) 
        pairId.(unique(x.resonances.pairId)) # try first if complex valued
    else
        resonanceId.(x.resonances.id) # function overload
    end
end


Chakra.pts(x::DRS)::Vector{Id} = sliceId.(unique(x.resonances.sliceId)) 

Chakra.pts(x::FUND)::Vector{Id} = fundamentalId.(unique(x.resonances.f0)) 
Chakra.pts(x::Fundamental)::Vector{Id} = resonanceId.(x.resonances.id)

Chakra.pts(x::REAL)::Vector{Id} = resonanceId.(x.resonances.id)

## Expansions basic structure with machine learning (clustering)
# Note: the complex values are removed from the dataset for clustering
Chakra.pts(x::HARM)::Vector{Id} = harmonicId.(unique(x.resonances.harmonic)) 
Chakra.pts(x::Harmonic)::Vector{Id} = resonanceId.(x.resonances.id)

# Groups of Slices and resonances
Chakra.pts(x::SliceSequence)::Vector{Id} = sliceId.(unique(x.resonances.sliceId))  



####################################  find a hierarchical structure by id ###########################################

Chakra.fnd(x::ResonanceId, m::Module) = begin
    i = findall(==(x.value),m.__data__.data.id)
    isempty(i) ? none : Resonance(m.__data__.data[i[1],:])
end


Chakra.fnd(x::PairId, m::Module) = begin
    # Should return all the elements from which it 
    #return vector of resonanceId's
    Pair(x, m.__data__)
end

Chakra.fnd(x::SliceId, m::Module) = begin
    Slice(x, m.__data__)
end


Chakra.fnd(x::FundamentalId, m::Module) = begin
    Fundamental(x, m.__fundamental__)
end

Chakra.fnd(x::HarmonicId, m::Module) = begin
    Harmonic(x, m.__harmonics__)
end


Chakra.fnd(x::DRSId, m::Module) = begin
    DRS(x, m.__data__)
    #Base.get(m.__data__.data.id,x,none)
end

Chakra.fnd(x::HARMId, m::Module) = begin
    HARM(x, m.__harmonics__)
end

Chakra.fnd(x::FUNDId, m::Module) = begin
    FUND(x, m.__fundamental__)
end

Chakra.fnd(x::REALId, m::Module) = begin
    REAL(x, m.__real__)
end

end
