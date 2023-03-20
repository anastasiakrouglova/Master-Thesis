# Load installed libraries
import pandas as pd
import os
import sys

# Load local libraries
# __file__ = "utilities_store.py"
# ROOT = './fpt/'

# # Add path to own library
# path = os.path.abspath(os.path.join(os.path.dirname(__file__), ROOT))
# if not path in sys.path:
#     sys.path.insert(1, path) # Warning: can probably cause errors on other machines since paths were not deleted, run print(sys.path) to examine which paths were created

# import utilities_store as us

##########################################################################################

DATA = "./fpt/data/output/flute-a4.csv"

# Load csv file of raw data
raw = pd.read_csv(DATA)

# print(os.path.abspath(DATA))

def removeNoiseFromPadding(data):
    # Noise appears at the beginning and end of the data from padding the signal 
    # with zeros on the left and right so that the window/step-size/overlap work out.
    return data[(data.onset != 0) & (data.onset != max(data.onset))]

