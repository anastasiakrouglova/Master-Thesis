using CSV, DataFrames

df = DataFrame(CSV.File("./fpt/data/output/filtered-clustered-C5F4.csv"))


df[!,:harmonics] = 0


for i in 1:maximum(df.dynamicResonance)
counter = 0
# foreach combination of clusters
cluster1 = df[df.dynamicResonance .== 2, :]
cluster2 = df[df.dynamicResonance .== 3, :]

# avg onset:
describe(cluster1.onset)
describe(cluster2.onset)

if (abs(mean(cluster1.onset) - mean(cluster2.onset)) <=  5000)
    print(abs(mean(cluster1.onset) - mean(cluster2.onset)))
    print("they are an harmonical cluster")
    counter = counter + 1
end

cluster1[!,:harmonics] = counter
cluster2[!,:harmonics] = counter
