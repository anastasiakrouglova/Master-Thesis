# This file runs the Fast Padé transform (FPT), 
# extract the fundamental and saves both files as a CSV in `fpt/data/output` together with the DRS

# Run the clustering algorithm written in Julia on the saved CSV's for the c_<file>.csv 
# and test the hierarchical knowledge representation in knowledge_heararchy.ipynb

import os, sys
from os.path import exists
from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path  
import pandas as pd

## 0. Import the library of functions
sys.path.insert(0, './fpt')
from utilities_store import *
import serialization as ser


filename = 'violin_canonD_1'
path_input = 'fpt/data/input/scores/' #scores/
path_output = 'fpt/data/output/scores/'

file = path_output+filename+".csv"

file_exists = exists(file)


if file_exists:
    # 1. Obtain the already generated csv fro mthe spectrogram
    df = pd.read_csv(file)
    print(df)

else:
    # 1. Obtain spectrogram of the signal
    spectrogram, signal = get_spectrogram(path_input+filename+'.wav', N = 300, step_size = 250, 
                                            power_threshold = 1e-9, amp_threshold = 1e-8)

    ## 2. Save the spectrogram as a csv file (and png)
    plot_spectrogram(spectrogram, max_freq = 2000, scale=100)
    ser.to_csv(spectrogram, filename, path_output)
    plt.savefig(path_output+filename+"_base")

    ## 3. Extract the fundamental frequencies (+ save plot)
    f0 = get_f0(spectrogram)
    plot_f0(spectrogram, f0, max_freq = 2000)
    plt.savefig(path_output+filename)
    # plt.savefig(path_output+filename+'.svg', format='svg', dpi=1200) # for thesis figures

    df = pd.read_csv(path_output+filename+'.csv')  
    
    # if (df['frequency'].isin(f0[0])):
    #     df['f0'][i] = 1
    # else:
    #     df['f0'][i] = 0
        
    # df_f0 = df[df['frequency'].isin(f0[0])]
    # print(df_f0)
    #df['f0'] = df_f0

    # filepath = Path(path_output+filename+'_f0.csv')  
    # filepath.parent.mkdir(parents=True, exist_ok=True)  
    # df_f0.to_csv(filepath)  


