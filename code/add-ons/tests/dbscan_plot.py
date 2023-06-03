import numpy as numpy
import scipy as scipy
from sklearn import cluster
import matplotlib.pyplot as plt

#converts a numpy array to a list
def set_to_list(numpy_array):
	List = []
	for item in numpy_array:
		List.append(item.tolist())
	return List

#generate datasets
def GenerateData():
	x1=numpy.random.randn(50,2)
	x2x=numpy.random.randn(80,1)+12
	x2y=numpy.random.randn(80,1)
	x2=numpy.column_stack((x2x,x2y))
	x3=numpy.random.randn(100,2)+8
	x4=numpy.random.randn(120,2)+15
	z=numpy.concatenate((x1,x2,x3,x4))
	return z

def DBSCAN(dataset, eps,MinPts,DistanceMethod = 'euclidean'):
#    Dataset is a mxn matrix, m is number of item and n is the dimension of data
	m,n=dataset.shape
	Visited=numpy.zeros(m,'int') #stores whether a point is visited or not
	Type=numpy.zeros(m) #it stores the type of a point 
#   -1 noise, outlier
#    0 border
#    1 core
	ClustersList=[] #set of clusters
	Cluster=[]
	PointClusterNumber=numpy.zeros(m) #stores which point belongs to which cluster
	PointClusterNumberIndex=1 #here each cluster is represented by some indexes starting from 1
	PointNeighbors=[] #stores neighbouring point of a given point
	DistanceMatrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(dataset, DistanceMethod))
	for i in xrange(m):
		if Visited[i]==0:
			Visited[i]=1
		PointNeighbors=numpy.where(DistanceMatrix[i] & lt;eps)[0]
			if len(PointNeighbors)&lt;MinPts:
				Type[i]=-1
			else:
				for k in xrange(len(Cluster)):
					Cluster.pop()
				Cluster.append(i)
				PointClusterNumber[i]=PointClusterNumberIndex
				PointNeighbors=set2List(PointNeighbors)    
				expandCluster(dataset[i],PointNeighbors,Cluster,MinPts,eps,Visited,DistanceMatrix,PointClusterNumber,PointClusterNumberIndex)
				Cluster.append(PointNeighbors[:])
				ClustersList.append(Cluster[:])
				PointClusterNumberIndex=PointClusterNumberIndex+1               
	return PointClusterNumber 





def expandClsuter(PointToExpand, PointNeighbors,Cluster,MinPts,eps,Visited,DistanceMatrix,PointClusterNumber,PointClusterNumberIndex  ):
	Neighbors=[] 
	for i in PointNeighbors:
		if Visited[i]==0:
			Visited[i]=1
			Neighbors=numpy.where(DistanceMatrix[i]=MinPts:
				# Neighbors merge with PointNeighbors
				for j in Neighbors:
					try:
						PointNeighbors.index(j)
					except ValueError:
						PointNeighbors.append(j)
                    
		if PointClusterNumber[i]==0:
			Cluster.append(i)
			PointClusterNumber[i]=PointClusterNumberIndex
	return


Data=GenerateData()
 
#Adding some noise with uniform distribution 
#X between [-3,17],
#Y between [-3,17]
noise=scipy.rand(50,2)*20 -3
 
Noisy_Data=numpy.concatenate((Data,noise))
size=20
 
 
fig = plt.figure()
ax1=fig.add_subplot(2,1,1) #row, column, figure number
ax2 = fig.add_subplot(212)
 
ax1.scatter(Data[:,0],Data[:,1], alpha =  0.5 )
ax1.scatter(noise[:,0],noise[:,1],color='red' ,alpha =  0.5)
ax2.scatter(noise[:,0],noise[:,1],color='red' ,alpha =  0.5)
 
 
Epsilon=1
MinumumPoints=20
result =DBSCAN(Data,Epsilon,MinumumPoints)
 

print(result)

 
for i in xrange(len(result)):
	ax2.scatter(Noisy_Data[i][0],Noisy_Data[i][1],color='yellow' ,alpha =  0.5)
      
plt.show()