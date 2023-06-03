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

struct HarmonicId <: Id
    value::Int
end

struct FundamentalId <: Id
    value::Int
end

struct DRSId <: Id
    value::Int
end

struct HARMId <: Id
    value::Int
end

struct FUNDId <: Id
    value::Int
end

struct REALId <: Id
    value::Int
end

# Call
resonanceId(i::Int) = ResonanceId(i)
pairId(i::Int) = PairId(i)

sliceId(i::Int) = SliceId(i)
harmonicId(i::Int) = HarmonicId(i)
fundamentalId(i::Int) = FundamentalId(i)

DRSid(i::Int) = DRSId(i) # it is possible to have multiple DRS's of 1 audiofragment: eg with different framewidths!!!
HARMid(i::Int) = HARMId(i)
FUNDid(i::Int) = FUNDId(i)
REALid(i::Int) = REALId(i)

sliceSeqId(i::Int) = SliceSequenceId(i) #other dimension: DRS points directly to sliceId, not really implemented

