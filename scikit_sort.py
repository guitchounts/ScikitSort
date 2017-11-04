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

color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])


def plot_results(X, Y_, means, covariances, index, title):
    plt.figure(figsize=(20,20))  #
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim([-2000,2000])
    plt.ylim([-1000,1000])
    #plt.xlim(-9., 5.)
    #plt.ylim(-3., 6.)
    #plt.xticks(())
    #plt.yticks(())
    plt.title(title)


def get_spikes_times(spikes_path,times_path,stim_data=None):
    print '############################## Reading in Spikes Files ##############################'

    #times_path = '/Users/guitchounts1/Documents/Ephys_Data/GRat21/636213969321601134/ChGroup_0/SpikeTimes'

    spiketimes_file = open(times_path,"rb")

    spiketimes = np.fromfile(spiketimes_file,dtype=np.uint64)

    spiketimes_file.close


    #spikes_path = '/Users/guitchounts1/Documents/Ephys_Data/GRat21/636213969321601134/ChGroup_0/Spikes'

    spikes_file = open(spikes_path,"rb")

    spikes = np.fromfile(spikes_file,dtype=np.int16)

    spikes_file.close

    spikes = np.reshape(spikes,[4,64,len(spikes)/(4*64)],'F') # reshape to 4 x 64 x numspikes

    ######## want to just take the spikes from the behavior session? 
    if stim_data != None:
        start_time = stim_data.times[0]
        end_time = stim_data.times.irow(-1) # for some reason just [-1] doesn't work...
        behaviortime_boundaries = [np.ceil(x*3e4) for x in [start_time-1,end_time+1]] ### the boundaries are in samples b/c spiketimes are in samples...
        trial_range = 30 # let's take spikestimes in the e.g. 30 sec before the first stim and 30 sec after the last.
        #behaviortime_boundaries = [start_time-trial_range,end_time+trial_range]
        print '################################# behavior time boundaries = ', behaviortime_boundaries
        ndx = [np.where(spiketimes>=behaviortime_boundaries[0])[0].min(),np.where(spiketimes<=behaviortime_boundaries[1])[0].max()]
        spiketimes = spiketimes[ndx[0]:ndx[1]+1] # +1 to include that last index
        print 'spiketimes[0] and spiketimes[-1] = ', spiketimes[0], spiketimes[-1]
        spikes = spikes[:,:,ndx[0]:ndx[1]+1]
        print '################################# Found a total of %d spikes #################################' % len(spiketimes)
    else:
        print 'Stim_Data not given'
        


    spike_vec = np.reshape(spikes,[4*64,spikes.shape[2]]) # reshape to 256 x numspikes

    spikes = None


	

    return spike_vec, spiketimes

def gmm_bic(features,save_path):
    print '############################## STARTING GMM ##############################'


    ## features should be shaped observations x features. Check to make this is the case:
    if features.shape[0] < features.shape[1]:
        features = features.T

    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, 20)
    cv_types = ['full'] # 'spherical', 'tied', 'diag', 
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = mixture.GaussianMixture(n_components=n_components,
                                          covariance_type=cv_type,max_iter=5000)
            gmm.fit(features)
            bic.append(gmm.bic(features))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm

    bic = np.array(bic)
    color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                                  'darkorange','green','magenta','yellow','red','purple'])
    clf = best_gmm
    bars = []
    print '############################## PLOTTING BIC scores ##############################'

        # Plot the BIC scores
    plt.figure(figsize=(20,20))
    spl = plt.subplot(2, 1, 1)
    for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
        xpos = np.array(n_components_range) + .2 * (i - 2)
        bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                      (i + 1) * len(n_components_range)],
                            width=.2, color=color))
    plt.xticks(n_components_range)
    plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
    plt.title('BIC score per model')
    xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
        .2 * np.floor(bic.argmin() / len(n_components_range))
    plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
    spl.set_xlabel('Number of components')
    spl.legend([b[0] for b in bars], cv_types)

    # Plot the winner
    splot = plt.subplot(2, 1, 2)
    Y_ = clf.predict(features)
    
    for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_,
                                               color_iter)):
        v, w = linalg.eigh(cov)
        if not np.any(Y_ == i):
            continue
        plt.scatter(features[Y_ == i, 0], features[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan2(w[0][1], w[0][0])
        angle = 180. * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(.5)
        splot.add_artist(ell)
    print '############################## Found %d Clusters!! ##############################' % (i+1)

    #plt.xticks(())
    #plt.yticks(())
    plt.xlim([-5000,5000])
    plt.ylim([-5000,5000])
    plt.title('Selected GMM: full model, 2 components')
    plt.subplots_adjust(hspace=.35, bottom=.02)
    #plt.show()
    plt.savefig(save_path + '/BIC_Results.pdf')


    return Y_

def plot_tetrode(spikes,cluster_assignments,save_path):
    print '############################## PLOTTING CLUSTERS ##############################'
    clusters = range(cluster_assignments.clusters.max()+1)

    #clust_spikes = []
    for clust in clusters:

        temp_spikes = np.squeeze(spikes[:,np.where(cluster_assignments.clusters==clust)])
        #clust_spikes.append(temp_spikes) ### this will then be a list of len=num_clusters, with each entry = array of spikes.
        #temp_spikes = None


        fig = plt.figure(figsize=(20, 10)) 

        gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1]) 
        
        ################## PLOT the mean waveforms #######################
        ax1 = sns.set_style("white")
        ax1 = plt.subplot(gs[0])
        num_samples =temp_spikes.shape[0]
        time_vec = range(num_samples)

        num_spikes = temp_spikes.shape[1]

        win_avg = np.mean(temp_spikes,axis=1)
        win_std = np.std(temp_spikes,axis=1)
        
        
        ax1 = errorfill(time_vec,win_avg,win_std,ax=ax1)
        plt.xlim([0,temp_spikes.shape[0]])
        plt.title('%d spikes in cluster %d' % (num_spikes,clust))
        
        ################## PLOT the waveform histogram #######################
        ax2 = plt.subplot(gs[1],sharex=ax1)
        voltmin = np.percentile(temp_spikes,1)-10
        voltmax = np.percentile(temp_spikes,99)+10
        xedges = np.arange(0.5,len(time_vec)+0.5,1)
        yedges = np.linspace(voltmin,voltmax,200)


        coordmat =  np.ravel(temp_spikes.T)
        coord_time = np.tile(range(len(time_vec)),temp_spikes.shape[1])

        #density,xedges,yedges = np.histogram2d(y=coord_time,x=coordmat,bins=(yedges,xedges))
        #ax2 = plt.imshow(density,cmap='gnuplot2',origin='lower')
        #################### ax2 = plt.pcolormesh(density,cmap='gnuplot2')
        for i in range(0,num_spikes,100):
            plt.plot(coord_time[0:256],temp_spikes[:,i],color='k',alpha=0.1)
        plt.xlim([0,temp_spikes.shape[0]])

        ################## PLOT the ISI #######################
        ax3 = plt.subplot(gs[2])
        
        clu_times = cluster_assignments.times[cluster_assignments.clusters==clust]
        clu_times = np.sort(clu_times)
        clu_isi = np.diff(clu_times)*1e3 ## convert from seconds to ms

        n, bins, patches = plt.hist(clu_isi,bins=1000,range=[0.1,100],histtype='stepfilled',log=True)
        plt.setp(patches, 'facecolor', 'magenta', 'alpha', 0.5)
        plt.xscale('log')
        plt.xlim([0.1, 100])
        plt.xlabel('ISI (ms)')
        violations = np.float(len(np.where(clu_isi<=1.0)[0])) / np.float(len(clu_isi)) * 100
        plt.title('Violations: %f%% <1ms' % (violations))
        

        ################## save fig: ####################
        sns.despine(ax=ax1)
        sns.despine(ax=ax3)

        #fig.tight_layout()
        print '############################## Saving Waveform of Cluster %d  ##############################' % clust
        fig.savefig(save_path + '/waveforms_cluster_' + str(clust) + '.pdf')


def errorfill(x, y, yerr, color=None, alpha_fill=0.3, ax=None):
    ax = ax if ax is not None else plt.gca()
    if color is None:
        #color = ax._get_lines.prop_cycler.next()
        #color= color['color']
        color = 'b'
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr

    ax.plot(x, y, color='k')
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)


def run_sorting(spikes_path,times_path,stim_data,save_path=os.getcwd()):

    spikes,spiketimes = get_spikes_times(spikes_path,times_path,stim_data)
    

    #if len(sys.argv) == 4:
    print '############################## Running Dimensionality Reduction ##############################'
    reduced_data = PCA(n_components=4).fit_transform(spikes.T)

    cluster_results = gmm_bic(reduced_data,save_path)

    # save cluster results with spike times:
    d = dict(times = spiketimes/3e4,clusters=cluster_results)
    cluster_assignments = pd.DataFrame.from_dict(d)
    cluster_assignments.to_csv(save_path + '/cluster_assignments.csv')


    ## plot cluster waveforms:
    plot_tetrode(spikes,cluster_assignments,save_path)


if __name__ == "__main__":

################################ take spikes and spiketimes files, get features and sort using Scikit Learn's GMM library.


    ## give this either 1) spikes, spike_times, and stim_data.csv paths as inputs, or 2) the cluster_assignments.csv 
    spikes_path = sys.argv[1] #
    times_path = sys.argv[2] #
    
    if len(sys.argv) > 3:
        stim_data_file = sys.argv[3]
        if stim_data_file.find('csv') == -1: 
            stim_data = pd.read_pickle(sys.argv[3]) # the behavior data - for first and last stim times.
        else:
            stim_data = pd.read_csv(sys.argv[3])
    else:
        stim_data = None

    run_sorting(spikes_path,times_path,stim_data)




