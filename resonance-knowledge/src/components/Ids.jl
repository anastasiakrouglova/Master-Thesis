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

# A dynamic resonance is a horizontal group of resonances, mostly grouped by pitch
struct HarmonicId <: Id
    value::Int
end

struct DRSId <: Id
    value::Int
end

struct DynRId <: Id
    value::Int
end

# Call
resId(i::Int) = ResonanceId(i)
pairId(i::Int) = PairId(i)
sliceId(i::Int) = SliceId(i)
drsId(i::Int) = DRSId(i) # it is possible to have multiple DRS's of 1 audiofragment: eg with different framewidths!!!
dynRId(i::Int) = DynRId(i)
harmonicId(i::Int) = HarmonicId(i)

sliceSeqId(i::Int) = SliceSequenceId(i) #other dimension: DRS points directly to sliceId, not really implemented

