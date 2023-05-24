using CSV, DataFrames, GLM, MLBase, Statistics
using Lathe.preprocess: TrainTestSplit

filename = "violin_canonD_1_f0"
PATH = "./fpt/data/output/scores/clustered/" * filename * ".csv"

G = "./score_conversion/frequenciesToNote.csv"
df = DataFrame(CSV.File(PATH))
notes = DataFrame(CSV.File(G))


function getFrequency(i)
    cluster = df[df.clusters .== i, :]
    vec_cluster = collect(cluster.frequency)
    avg_cluster = mean(vec_cluster)
    frequency_pred = round(avg_cluster, digits = 2)
    println(frequency_pred)

    return frequency_pred
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

for i in 1:maximum(df.clusters)
    freq = getFrequency(i)
    noteName = getNotename(freq)
    push!(frequencies, freq)
end

all_notes = Vector{String}()

# return note
for i in 1:length(frequencies)
    note = getNotename(frequencies[i])
    print(note)
    push!(all_notes, note)
    
end

n = join(all_notes)
println(n)

# save notes in a textfile -> send to text_to_music
file = open("score_conversion/notes.txt", "w")
write(file, n)
close(file)