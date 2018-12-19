import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from os.path import join, isfile
# -------------------------------------------------------------------------------------
from nilearn import plotting, image;
from nilearn.masking import compute_epi_mask
import nibabel as nib
from nilearn.input_data import NiftiMasker
import time
# -------------------------------------------------------------------------------------
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.cross_validation import LeaveOneLabelOut, cross_val_score
from sklearn.feature_selection import SelectPercentile, f_classif

PATH = '/imaging/ourlab/jerez/memcon/'

def apply_niftiMasker(subject,runs):
    """Compute a time-series matrix for an entire scanning session.

    Args:
        subject: subject number (i.e. 'S01')
        runs: list of runs

    Returns:
        None
    """

    for run in runs:
        print run
        func = join(PATH,'output_firstSteps','smoothed','run' +str(run) +'_' +subject,'swa' +subject +'_' +'run' +str(run) +'.nii')
        onsets = 'onsets/' +subject +'_run' +str(run) +'_videos.txt'

        print func

        # define mask from run 1
        epi_masker = NiftiMasker(mask_strategy='epi')
        epi_masker.fit('/imaging/ourlab/jerez/memcon/output_firstSteps/smoothed/run1_' +subject +'/swa' +subject +'_run1.nii')

        start = time.time()
        masker = NiftiMasker(mask_img=epi_masker.mask_img_, standardize=True, memory="nilearn_cache", memory_level=1)
        X = masker.fit_transform(func)
        end = time.time()
        print 'time = ' +str((end-start))

        print X.shape # (total TRs x number of voxels per volume)
        plotting.plot_roi(epi_masker.mask_img_,image.mean_img(func), title='EPI automatic mask')

        # get onsets
        df = pd.read_csv(join(PATH,onsets),sep='\s',header=None)
        df.columns = ['onset','duration','category'] # df.shape is (16 x 3)

        x = np.ndarray(shape=(df.shape[0],X.shape[1]))
        x.shape # (16 x number of voxels per volume)

        for onset in range(df.shape[0]):
            i = df.onset[onset]
            print i,i+30
            x[onset,:] = X[int(i):int(i+30)].mean(axis=0)

        if run == 1:
            Z = x
        else:
            Z = np.concatenate((Z,x))
        print Z.shape
        np.save(join(PATH,'data/Z_files',subject +'_Z.npy'),Z)
        
def load_scanning_session_matrix(subject):
    """ Load a scanning session's time-series matrix

    Args:
        subject: subject number (i.e. 'S01')

    Returns:
        a subject's scanning session time-series matrix
    """
    Z = np.load(join(PATH,'data/Z_files',subject + '_Z.npy'))
    print Z.shape
    return Z

def get_onsets(subject,runs):
    """Compute a time-series matrix for an entire scanning session.

    Args:
        subject: subject number (i.e. 'S01')
        runs: list of runs

    Returns:
        None
    """
    df_onsets = pd.DataFrame()
    for run in runs:
        print run,
        func = join(PATH,'output_firstSteps','smoothed','run' +str(run) +'_' +subject,'swa' +subject +'_' +'run' +str(run) +'.nii')
        onsets = 'onsets/' +subject +'_run' +str(run) +'_videos.txt'

        # get onsets
        df = pd.read_csv(join(PATH,onsets),sep='\s',header=None)
        df.columns = ['onset','duration','category']
        df_onsets = df_onsets.append([df])
    df_onsets = df_onsets.reset_index(drop=True)
    print df_onsets.head()
    return df_onsets

def plot_classification_accuracies(df):
    
    import scipy as sp
    print sp.stats.ttest_rel(df['classification_accuracy'],df['classification_accuracy_reshuffled'])
    
    n_subjects = df.shape[0]
    
    newdf = pd.concat([pd.Series(['classification_accuracy']*n_subjects),df['classification_accuracy']],axis=1,ignore_index=True)
    q = pd.concat([pd.Series(['classification_accuracy_reshuffled']*n_subjects),df['classification_accuracy_reshuffled']],axis=1,ignore_index=True)
    newdf = newdf.append([q],ignore_index=True)

    newdf.columns = ['temp','accuracy']
    ax1 = sns.swarmplot(x = 'temp',y = 'accuracy',data = newdf,size=8)
    ax2 = sns.boxplot(x = 'temp',y = 'accuracy',data = newdf,
            showcaps=False,boxprops={'facecolor':'None'},
            showfliers=False,whiskerprops={'linewidth':0})
    ax1.set_xlabel('')
    ax1.tick_params(axis='both', labelsize=14)
    ax1.set_ylabel('Accuracy',fontsize=14)
