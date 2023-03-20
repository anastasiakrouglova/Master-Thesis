from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import numpy as np

# STILL IN PROCESS

def DBSCAN(data, eps, min_samples,  max_frequency): # good max_frequency = 2000
    # Preprocessing
    freq = data.frequency[(0 < data.frequency) & (data.frequency < max_frequency)]
    power = data.power

    df = pd.DataFrame({'power':power, 'frequency':freq})
    filtered_df = df[df['frequency'].notnull()]
    
    X = filtered_df.to_numpy()
    X = StandardScaler().fit_transform(X)
    
    # DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)
    
    
    
    
    
    
def hyperParameterTuning(X, range_n_clusters):
    # hyperparameter tuning depending on data
    #range_n_clusters = [3, 4, 5] # range_n_clusters in variable

    for n_clusters in range_n_clusters:
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)
        
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("for n_clusters=", n_clusters,
            "The average silhouette_score is :", silhouette_avg)
        
        # compute the silhouette score for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)



        
        



min_samples = [1,10, 15, 17, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 79, 75, 80]
for i in min_samples:
    print("min_samples value is "+str(i))
    db = DBSCAN(eps=0.5, min_samples=i).fit(powFreq)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    # ignoring the label '-1' as its for outliers
    labels = set([label for label in db.labels_ if label >= 0])
    print(set(labels))
    print("For min_samples value ="+str(i), "Total no. of clusters are "+str(len(set(labels))))       
        
        
# Find min_amount of samples        
min_samples = [1,10, 15, 17, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 79, 75, 80]
for i in min_samples:
    print("min_samples value is "+str(i))
    db = DBSCAN(eps=0.5, min_samples=i).fit(powFreq)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    # ignoring the label '-1' as its for outliers
    labels = set([label for label in db.labels_ if label >= 0])
    print(set(labels))
    print("For min_samples value ="+str(i), "Total no. of clusters are "+str(len(set(labels))))        


def findOptimalEpsilon():
    # Find best EPS value
    range_eps = [0.1, 0.2, 0.3, 0.4, 0.5]
    for i in range_eps :
        print("eps value is "+str(i))
        db = DBSCAN(eps=i, min_samples=5).fit(powFreq)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        print(set(labels))
        print(set(labels))
        silhouette_avg = silhouette_score(powFreq, labels)
        print("For eps value ="+str(i), #labels,
            "The average silhouette_score is :", silhouette_avg)
        