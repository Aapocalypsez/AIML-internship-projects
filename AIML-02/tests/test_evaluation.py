import numpy as np
from ml.strong_evaluation import patient_interval,ece


def test_bootstrap_groups_repeated_patient_images_and_degenerate_samples():
    labels=np.array([0,0,1,1]); scores=np.array([.1,.2,.8,.9]); patients=np.array(['a','a','b','b'])
    interval=patient_interval(labels,scores,patients,repeats=100)
    assert interval['lower']==interval['upper']==1.
    assert 0<interval['valid_replicates']<100


def test_calibration_error_perfect_and_confidently_wrong():
    y=np.array([0,1]); assert ece(y,np.array([0.,1.]))==0.
    assert ece(y,np.array([1.,0.]))==1.
