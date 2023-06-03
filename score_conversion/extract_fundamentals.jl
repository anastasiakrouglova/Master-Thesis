using CSV, DataFrames, GLM, MLBase, Statistics
using Lathe.preprocess: TrainTestSplit

# filename = "violin_canonD_1"
filename = "flute_syrinx_artificial_1"
PATH = "./fpt/data/output/scores/clustered/" * filename * ".csv"

Sharp = "./score_conversion/frequenciesToNoteSharp.csv"
Flat = "./score_conversion/frequenciesToNoteFlat.csv"

df = DataFrame(CSV.File(PATH))
notes = DataFrame(CSV.File(Flat))

# Time signature: manual features for nrow
beats_per_measure = 3
note_one_beat = 4

amount_measures = 1

function getFrequency(i)
    cluster = df[df.f0 .== i, :]
    vec_cluster = collect(cluster.frequency)
    avg_cluster = mean(vec_cluster)
    frequency_pred = round(avg_cluster, digits = 2)
    #println(frequency_pred)

    return frequency_pred
end

function getonset(i)
    cluster = df[df.f0 .== i, :]
    onset = minimum(cluster.onset)

    return onset
end

function getduration(i)
    cluster = df[df.f0 .== i, :]
    onset = minimum(cluster.onset)
    offset = maximum(cluster.onset)
    duration = offset - onset

    return duration
end

function getNotename(freq)
    minDist = 1000
    notename = "C"

    for i in 1:nrow(notes)
        if (abs((notes.Frequency[i] - freq)) < minDist)
            minDist = abs((notes.Frequency[i] - freq))
            notename = notes.Notename[i]
        end
    end

    return notename
end


frequencies = Vector{Float64}()
noteNames = Vector{String}()
noteDurations = Vector{String}()

min_clus = 1
max_clus = maximum(df.f0)

minRes = minimum(df[df.f0 .== min_clus, :onset])
maxRes = maximum(df[df.f0 .== max_clus, :onset])


# Note: we assume that the fragment does not end or start with a rest
lengthFragment = maxRes-minRes

# Possible lengths of notes
note_lengths = [1, 2, 3, 4, 6, 8, 12, 16, 32, 64, 128]  # Als geen power van 2 of multiplicatie van 6

for i in 1:maximum(df.f0)
    freq = getFrequency(i)
    duration = getduration(i)
    duration = (duration / lengthFragment) * (beats_per_measure * amount_measures)
    note_length = (1 / duration) * beats_per_measure

    minDistb = 1000
    noteDur = 100
    l = length(note_lengths)
    for i in 1:l
        if (abs((note_lengths[i] - note_length)) < minDistb)
            minDistb = abs((note_lengths[i] - note_length))
            noteDur = note_lengths[i]
        end
    end

    noteName = getNotename(freq)
    cluster = df[df.f0 .== i, :]

    # TODO:  add rests to musical notation (Future work)

    # artificial boundery for the thesis demonstration with Syrinx, needs more expanded rule-based approach
    if (minimum(cluster.frequency) > 540) 
        push!(frequencies, freq)

        if (ispow2(noteDur))
            push!(noteDurations, string(noteDur))
        else 
            # calculate the position of the last set bit of `n`
            lg = ceil(Int, log(noteDur)/log(2))
            # next power of two will have a bit set at position `lg`.
            pow2 = 1 << lg

            push!(noteDurations, string(pow2)*".")
        end
    end
end

all_notes = Vector{String}()

# return note
for i in 1:length(frequencies)
    note = getNotename(frequencies[i])
    push!(all_notes, note)  
end

res = [(n * d) for (n, d) in zip(all_notes, noteDurations)]
println(res)

beat = "\\time" * string(beats_per_measure) * "/" * string(note_one_beat)
res = beat * join(res)

# save notes in a textfile -> send to text_to_msusic
file = open("score_conversion/notes.txt", "w")
write(file, res)
close(file)