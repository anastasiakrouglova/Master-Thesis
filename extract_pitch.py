# Load installed libraries
import pandas as pd
import os
import sys
from pathlib import Path  

# Load local libraries
__file__ = "data_preprocessing.py"
ROOT = './ML-pipeline/'

# Add path to own library
path = os.path.abspath(os.path.join(os.path.dirname(__file__), ROOT))
if not path in sys.path:
    sys.path.insert(1, path) # Warning: can probably cause errors on other machines since paths were not deleted, run print(sys.path) to examine which paths were created
print(sys.path)

import data_preprocessing as dp
import DBSCAN as fe
import linear_regression as lr
#print(sys.modules)
#print(dir(dp))


###########################################################################################################################
# TODO: Generate CSV file of a sound (dp.py)

# Load data
DATA = "./fpt/data/output/flute-a4.csv"
MAX_FREQ = 2000

# Denoise raw data
raw = pd.read_csv(DATA)
data = dp.remove_noise(raw, 0.001)

# Run DBSCAN 
def extract_pitch_file():
    labels, n_clusters_ = fe.cluster_DBSCAN(data, MAX_FREQ)
    print(labels)

    # insert labels into dataframe filtered on MAX_FREQ
    data_with_clusters = fe.insert_cluster_data(data, MAX_FREQ, labels)

    filepath = Path('./data-ml/out.csv')  
    filepath.parent.mkdir(parents=True, exist_ok=True)  
    data_with_clusters.to_csv(filepath)  

    # Find intercept of every cluster with linear regression
    predictions = [] 
    for i in range(n_clusters_): # unique(labels)
        predictions.append(lr.regression(data_with_clusters, i))
        
    print(predictions)
    
    return predictions # returns eg [440, 880, 1230]


extract_pitch_file()