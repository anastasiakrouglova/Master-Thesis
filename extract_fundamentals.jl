using CSV, DataFrames, GLM, MLBase, Statistics
using Lathe.preprocess: TrainTestSplit

#df = DataFrame(CSV.File("./fpt/data/output/filtered-clustered-C5F4.csv"))
df = DataFrame(CSV.File("./fpt/data/output/filtered-clustered-flute_a4.csv"))
notes = DataFrame(CSV.File("./frequenciesToNote.csv"))

function getFrequency(i)
    cluster1 = df[df.dynamicResonance .== i, :]
    train, test = TrainTestSplit(cluster1,.75)

    fm = @formula(frequency ~ onset)
    linearRegressor = lm(fm, train)

    # If standard deviation error < 0.05 and t big enough and r2 also big enough, r2(linearRegressor)
    slope = GLM.coef(linearRegressor)[1]
    stdError = GLM.coeftable(linearRegressor).cols[2][2]
    #pval = GLM.coeftable(linearRegressor).cols[4][2]
    #GLM.coeftable(linearRegressor)

    if (stdError < 0.05) # if statistically significant
        frequencyPred = round(slope, digits = 2)
    else 
        print("too much noise")
    end

    # frequency to note
    return frequencyPred
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

for i in 1:maximum(df.dynamicResonance)
    freq = getFrequency(i)
    noteName = getNotename(freq)
    push!(frequencies, freq)
end

frequencies

# filter out fundamentals

# 1. sorteren per tijd in dubbele array

# 2. In 1 tijdsfragment: haal de fundamental eruit
# todo: run multiple times, neem gemiddelde van frequencies
f0 = min(frequencies...)


# 3. akkoorden eruit leren halen


# return note
note1 = getNotename(f0)
n = convert(String,note1)

# sla noten op in een textfile -> stuur naar text_to_music

file = open("notes.txt", "w")
write(file, n)
close(file)