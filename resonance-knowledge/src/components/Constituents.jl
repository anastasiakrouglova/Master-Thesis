# Definition Constituents

struct Resonance <: Constituent
    data::DataFrameRow
    # TODO: instead of row: just id
end

# An abstract type to filter resonances
abstract type ResonanceSet end

# struct Resonance <: ResonanceSet
#     id::ResonanceId
#     data::DataFrameRow
#     print("aaaaaaa")
#     Resonance(id::ResonanceId, dataset::DRSHierarchy) = begin
#         df = filter(:id => i -> i == id.value, dataset.data)
#         print("bbbbbbb")
#         return isempty(df) ? none : new(id,df)
#     end
# end

struct ResonanceCollection <: ResonanceSet
    ids::Vector{ResonanceId}
    data::DataFrame
    ResonanceCollection(ids::Vector{ResonanceId}, dataset::DRSHierarchy) = begin
        values = [id.value for id in ids]

        df = filter(:id => o -> o in values, dataset.data)
        return isempty(df) ? none : new(ids, df)
    end
end

struct Pair <: ResonanceSet
    pairId::PairId
    resonances::DataFrame

    Pair(pairId::PairId, dataset::DRSHierarchy) = begin

        # if pair exists, filter, else throw error that dataset is not complex-valued
        
        df = filter(:pairId => p -> p == pairId.value, dataset.data)

        return isempty(df) ? none : new(pairId,df)
    end
end


###########################################################################################
#################################       Dynamic Resonance        ######################################
###########################################################################################

# A dynamic resonance is a horizontal group of resonances, mostly grouped by pitch
struct DynamicResonance <: ResonanceSet
    dynamicResId::DynamicResonanceId
    resonances::DataFrame
    DynamicResonance(dynamicResId::DynamicResonanceId,dataset::DRSHierarchy) = begin
        df = filter(:dynamicResonance => o -> o == dynamicResId.value, dataset.data)
        return isempty(df) ? none : new(dynamicResId,df)
    end
end


###########################################################################################
#################################       Slice        ######################################
###########################################################################################

# a slice in the spectrogram (based on time)
struct Slice <: ResonanceSet
    sliceId::SliceId
    resonances::DataFrame
    Slice(sliceId::SliceId,dataset::DRSHierarchy) = begin
        df = filter(:sliceId => o -> o == sliceId.value, dataset.data)
        return isempty(df) ? none : new(sliceId,df)
    end
end

struct SliceSequence <: ResonanceSet
    start::Union{SliceId,Colon}
    finish::Union{SliceId,Colon}
    resonances::DataFrame
    SliceSequence(start::Union{SliceId,Colon}, finish::Union{SliceId,Colon}, dataset::DRSHierarchy) = begin
        if (typeof(start) == SliceId && typeof(finish) == SliceId)
            df = filter(:sliceId => o -> start.value <= o <= finish.value, dataset.data)
        elseif (typeof(start) == SliceId && typeof(finish) == Colon)
            df = filter(:sliceId => o -> start.value <= o, dataset.data)
        elseif (typeof(start) == Colon && typeof(finish) == SliceId)
            df = filter(:sliceId => o -> o <= finish.value, dataset.data)
        end
        return isempty(df) ? none : new(start,finish, df)
    end
end

###########################################################################################
#################################     FREQUENCY      ######################################
###########################################################################################

struct FrequencyBand <: ResonanceSet
    min::Union{Real,Colon}
    max::Union{Real,Colon}
    resonances::DataFrame
    # frequency between [x, y]
    FrequencyBand(min::Union{Real,Colon}, max::Union{Real,Colon}, dataset::DRSHierarchy) = begin
        if (typeof(max) == Colon)
            df = filter(:frequency => f -> min <= f, dataset.data)
        elseif (typeof(min) == Colon)
            df = filter(:frequency => f -> f < max, dataset.data)
        else
            df = filter(:frequency => f -> min <= f <= max, dataset.data)
        end

        return isempty(df) ? none : new(min, max, df)
    end
    # Filter on pos, neg and 0 frequencies
    FrequencyBand(s::String, dataset::DRSHierarchy) = begin
        if (s == "pos")
            FrequencyBand(0, :, dataset::DRSHierarchy)
        elseif (s == "neg")
            FrequencyBand(:, 0, dataset::DRSHierarchy)
        elseif (s == "null")
            FrequencyBand(0, 0, dataset::DRSHierarchy)
        else
            error("please put in pos/neg/null")
        end
    end
end


struct DRS <: ResonanceSet
    id::DRSId # multiple DRSs of an audiofile are possible
    #resonances::DRSHierarchy
    resonances::DataFrame

    DRS(id::DRSId, dataset::DRSHierarchy) = begin

        # return collection of sliceId's
        return new(id, dataset.data)
       #return isempty(df) ? none : new(id, df)
    end 
end


