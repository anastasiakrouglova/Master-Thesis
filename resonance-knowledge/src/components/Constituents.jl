# Definition Constituents
struct Resonance <: Constituent
    data::DataFrameRow
end

# An abstract type to filter resonances
abstract type ResonanceSet end

struct ResonanceCollection <: ResonanceSet
    ids::Vector{ResonanceId}
    data::DataFrame
    ResonanceCollection(ids::Vector{ResonanceId}, dataset::DataSet) = begin
        values = [id.value for id in ids]

        df = filter(:id => o -> o in values, dataset.data)
        return isempty(df) ? none : new(ids, df)
    end
end

struct Pair <: ResonanceSet
    pairId::PairId
    resonances::DataFrame

    Pair(pairId::PairId, dataset::DataSet) = begin
        df = filter(:pairId => p -> p == pairId.value, dataset.data)

        return isempty(df) ? none : new(pairId,df)
    end
end



###########################################################################################
#################################       Slice        ######################################
###########################################################################################

# a slice in the spectrogram (based on time)
struct Slice <: ResonanceSet
    sliceId::SliceId
    resonances::DataFrame
    Slice(sliceId::SliceId,dataset::DataSet) = begin
        df = filter(:sliceId => o -> o == sliceId.value, dataset.data)
        return isempty(df) ? none : new(sliceId,df)
    end
end

struct SliceSequence <: ResonanceSet
    start::Union{SliceId,Colon}
    finish::Union{SliceId,Colon}
    resonances::DataFrame
    SliceSequence(start::Union{SliceId,Colon}, finish::Union{SliceId,Colon}, dataset::DataSet) = begin
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
    FrequencyBand(min::Union{Real,Colon}, max::Union{Real,Colon}, dataset::DataSet) = begin
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


struct DRS <: ResonanceSet
    resonances::DataSet

    DRS(df::DataSet) = begin
        return isempty(df) ? none : new(df)
    end 
end


