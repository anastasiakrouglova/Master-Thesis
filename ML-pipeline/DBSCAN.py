from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt


def cluster_DBSCAN(data, max_frequency): 
    # Empirical test for good max_frequency = 2000
    freq_pow_array_normalized, freq_pow_df = preprocessing_DBSCAN(data, max_frequency) 
    power_freq = freq_pow_df.sort_values(by=['power', 'frequency'])
    
    # Plot frequency — power
    plt.scatter(freq_pow_array_normalized[:, 0], freq_pow_array_normalized[:, 1])
    plt.show()
    
    # Hyperparameter tuning
    #eps = find_eps_value(power_freq, 25)
    #min_samples = 25 #find_min_samples(power_freq, 0.5) # TODO
    
    db = DBSCAN(eps=0.5, min_samples=35).fit(freq_pow_array_normalized)
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)
    
    plot_clustering_DBSCAN(freq_pow_array_normalized, db, labels, n_clusters_)
    
    return labels # labels of clusters
    

    

def plot_clustering_DBSCAN(X, db, labels, n_clusters_):
    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = labels == k

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=14,
        )

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=6,
        )

    plt.title(f"Estimated number of clusters: {n_clusters_}")
    plt.show()
    
    

def preprocessing_DBSCAN(data, max_frequency):
    # good max_frequency = 2000
    frequency = data.frequency[(0 < data.frequency) & (data.frequency < max_frequency)]
    power = data.power

    df = pd.DataFrame({'power':power, 'frequency':frequency})
    freq_pow_df = df[df['frequency'].notnull()]
    
    freq_pow_array = freq_pow_df.to_numpy()
    freq_pow_array_normalized = StandardScaler().fit_transform(freq_pow_array)
    
    return freq_pow_array_normalized, freq_pow_df

    

def find_eps_value(power_freq, min_samples):
    range_eps = [0.1, 0.2, 0.3, 0.4, 0.5]
    max_silhouette_avg = [0, 0]
    for i in range_eps :
        print("eps value is "+str(i))
        db = DBSCAN(eps=i, min_samples=min_samples).fit(power_freq)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        print(set(labels))
        silhouette_avg = silhouette_score(power_freq, labels)
        
        if(max_silhouette_avg[0] < silhouette_avg):
            max_silhouette_avg[0] = silhouette_avg
            max_silhouette_avg[1] = i
            
        print("For eps value ="+str(i), #labels,
            "The average silhouette_score is :", silhouette_avg)

    best_eps = max_silhouette_avg[1]
    
    return  best_eps

        
# hyperparameter tuning of min_samples
def find_min_samples(power_freq, eps):
    min_samples = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 79, 75, 80, 85, 90, 95, 100]
    
    for i in min_samples:
        print("min_samples value is "+str(i))
        db = DBSCAN(eps=eps, min_samples=i).fit(power_freq)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        # ignoring the label '-1' as its for outliers
        labels = set([label for label in db.labels_ if label >= 0])
        print(set(labels))
        print("For min_samples value ="+str(i), "Total no. of clusters are "+str(len(set(labels))))       
    
    #return max(...)
        


def insert_cluster_data(data, MAX_FREQ, labels):
    freq = data.frequency[(0 < data.frequency) & (data.frequency < MAX_FREQ)]

    df = pd.DataFrame({'onset':data.onset, 'duration':data.duration, 'sample_rate':data.sample_rate,
                            'amplitude':data.amplitude,	'phase':data.phase,
                            'frequency':freq,
                            'decay':data.decay, "power": data.power, 'd':data.d,	'w':data.w,	'z':data.z
                            })


    filtered_df = df[df['frequency'].notnull()]
    filtered_df.insert(2, "cluster", labels, True)
    
    return filtered_df
    
       