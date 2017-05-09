from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import datetime
import pickle
import sys

import numpy as np
import scipy as sp
import scipy.stats
import sklearn as sk
import sklearn.decomposition 
import sklearn.gaussian_process
import sklearn.model_selection
import sklearn.preprocessing 

# class Preprocess_None(x) :     
#     def fit_transform(self, x) :
#         return x
#     def fit(self, x) :
#         return x

def print_and_time(*args, **kwargs) : 
    now = datetime.datetime.utcnow()
    past = kwargs.pop('past', None)
    if not past is None  :
        elapsed = (now-past).seconds
        elapsed_min = elapsed // 60
        elapsed_sec = elapsed % 60
        print(" elapsed: %d'%d''" % (elapsed_min, elapsed_sec), file=kwargs.get('file', sys.stdout))
    end=kwargs.pop('end', '')
    print(*args, end=end, **kwargs)
    return now

def read_pickled_data(filename) :
    source = open(filename, 'rb')
    sizes = pickle.load(source)
    num_samples = sizes[0]
    feature_size = sizes[1]
    ids = []
    labels = np.empty([num_samples], dtype=np.float)
    features = np.empty([num_samples, feature_size], dtype=np.float)
    for s in xrange(num_samples) :
        sample = pickle.load(source)
        ids.append(sample[0])
        labels[s] = sample[1]
        features[s] = sample[2]
    source.close()
    return ids, labels, features

class exp2var() :
    def __init__(self, loc=0.0, scale=1.0) :
        self.dist = sp.stats.uniform(loc=loc, scale=scale)
        self.loc  = loc
        self.scale = scale
    def rvs(self, **kwargs) :
        u = self.dist.rvs(**kwargs)
        return 2.0**u

def new_classifier(linear=False, dual=True, max_iter=10000, min_gamma=-24, scale_gamma=8) :
    if linear :
        parameters = {
            'dual'         : [ dual ],
            'C'            : exp2var(loc=-16.0, scale=32.0),
            'multi_class'  : [ 'ovr' ], 
            'random_state' : [ 0 ], 
            'max_iter'     : [ max_iter ],
        }
        classifier = sk.svm.LinearSVC()
    else :
        parameters = {
            'C'                       : exp2var(loc=-16.0, scale=32.0),
            'gamma'                   : exp2var(loc=min_gamma, scale=scale_gamma),
            'kernel'                  : [ 'rbf' ], 
            'decision_function_shape' : [ 'ovr' ], 
            'random_state'            : [ 0 ],
        }
        classifier = sk.svm.SVC()
    return classifier, parameters

def hyperoptimizer(classifier, parameters, scoring='roc_auc', max_iter=10, n_jobs=1, group=True) :
    return sk.model_selection.RandomizedSearchCV(classifier, parameters, 
        n_iter=max_iter, scoring=scoring, fit_params=None, n_jobs=n_jobs, iid=True, refit=True, 
        cv=sk.model_selection.GroupKFold(n_splits=3) if group else None, 
        verbose=2, random_state=0)
