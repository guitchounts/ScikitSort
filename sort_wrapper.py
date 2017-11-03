import itertools
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec
import seaborn as sns
from sklearn import mixture
from sklearn.decomposition import PCA
import pandas as pd
import sys,os

#sys.path.append('/Users/guitchounts/Dropbox (coxlab)/Scripts/Repositories/continuous-ephys')
sys.path.append('Volumes/Mac HD/Dropbox (coxlab)/Scripts/Repositories/ScikitSort')
import scikit_sort


if __name__ == "__main__":

    ##### sort all channel groups:
    channels = [6] #range(16)

    stim_data = None
     ## give this either 1) spikes, spike_times, and stim_data.csv paths as inputs, or 2) the cluster_assignments.csv 
    
    experiment_path = sys.argv[1] ## e.g. /Volumes/steffenwolff/GRat31/636427282621202061/

    ch_group = 'ChGroup_'

    for ch in channels:

        spikes_path = experiment_path + ch_group + str(ch) + '/Spikes'
        times_path = experiment_path + ch_group + str(ch) + '/SpikeTimes'
        
        
        save_path = ch_group + str(ch)
        if not os.path.exists(save_path):
                os.makedirs(save_path)

        scikit_sort.run_sorting(spikes_path,times_path,stim_data,save_path)