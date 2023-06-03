using NearestNeighbors
using PyCall
np = pyimport("numpy")



data = rand(3, 10^4)
k = 3
point = rand(3)

kdtree = KDTree(data)
idxs, dists = knn(kdtree, point, k, true)
dists = np.sort(dists[:, :], axis=0)

# fig = plt.figure(figsize=(5, 5))
# plt.plot(distances)
# plt.xlabel("Points")
# plt.ylabel("Distance")

function linescatter1()
    trace2 = scatter(;x=[1, 2, 3], y=dists, mode="lines")
    plot([trace2])
end

linescatter1()


# idxs
# # 3-element Array{Int64,1}:
# #  4683
# #  6119
# #  3278

# dists
# # 3-element Array{Float64,1}:
# #  0.039032201026256215
# #  0.04134193711411951
# #  0.042974090446474184
