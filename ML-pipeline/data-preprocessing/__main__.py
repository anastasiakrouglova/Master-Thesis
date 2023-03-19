# Load installed libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys

##########################################################################################
__file__ = "utilities_store.py"
ROOT = "../../fpt/"
DATA = "../../fpt/data/output/flute-a4.csv"

# path resolving
path = os.path.abspath(os.path.join(os.path.dirname(__file__), ROOT))
if not path in sys.path:
    sys.path.insert(1, path)

import utilities_store as us
##########################################################################################

# Load csv file for flute
df = pd.read_csv(DATA)