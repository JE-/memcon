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

def compute_classification(Z,df_onsets,CV,percentile = None,randomize_labels = False):
    """Compute classification accuracy

    Args:
        Z: The 'Z' file for a given subject
        df_onsets: the onsets for the subject

    Returns:
        fitted model, Classification accuracy
    """
    if randomize_labels == True:
        from sklearn.utils import shuffle
        df_onsets = shuffle(df_onsets)
    svc = SVC(kernel='linear')

    # We have our classifier (SVC), our feature selection (SelectPercentile),and now,
    # we can plug them together in a *pipeline* that performs the two operations
    # successively:
    if percentile > 0:
        # Define the dimension reduction to be used.
        # Here we use a classical univariate feature selection based on F-test,
        # namely Anova. When doing full-brain analysis, it is better to use
        # SelectPercentile, keeping 5% of voxels
        # (because it is independent of the resolution of the data).
        feature_selection = SelectPercentile(f_classif, percentile=percentile)
        anova_svc = Pipeline([('anova', feature_selection), ('svc', svc)])
    else:
        anova_svc = Pipeline([('svc', svc)])

    #### fit a decoder and predict
    anova_svc.fit(Z, df_onsets['category'])
    y_pred = anova_svc.predict(Z)

    # Compute the prediction accuracy for the different folds (i.e. session)
    cv_scores = cross_val_score(anova_svc, Z, df_onsets['category'],cv=CV)

    # Return the corresponding mean prediction accuracy
    classification_accuracy = cv_scores.mean()

    # Print the results
    print("Classification accuracy: %.4f / Chance level: %f" %
          (classification_accuracy, 1. / len(np.unique(df_onsets['category']))))
    
    return (anova_svc,classification_accuracy)