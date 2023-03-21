# Load installed libraries
import pandas as pd
import os
import sys

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
#print(sys.modules)
#print(dir(dp))


###########################################################################################################################
# TODO: Generate CSV file of a sound (dp.py)

# Load data
DATA = "./fpt/data/output/flute-a4.csv"
raw = pd.read_csv(DATA)

# Denoise raw data
data = dp.remove_noise(raw, 0.001)

# Run DBSCAN
pre = fe.DBSCAN(data, 2000)
print(pre)