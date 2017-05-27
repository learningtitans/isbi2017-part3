# Copyright 2017 Eduardo Valle. All rights reserved.
# eduardovalle.com/ github.com/learningtitans
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
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

from svm_layer import utils as su

parser = argparse.ArgumentParser(prog='train_svm_layer.py', description='Train the SVM decision.')
parser.add_argument('--svm_method', type=str, default='RBF', help='svm method to employ: RBF (default), LINEAR_DUAL, or LINEAR_PRIMAL.')
parser.add_argument('--max_iter_svm', type=int, default=1000, help='maximum number of interations for the linear svm.')
parser.add_argument('--max_iter_hyper', type=int, default=10, help='maximum number of interations for the hyperparameter search.')
parser.add_argument('--jobs', type=int, default=1, help='number of parallel jobs in the hyperparameter search.')
parser.add_argument('--preprocess', type=str, default='PCA', help='train and apply a preprocessor for the data: PCA, PCA_WHITEN, Z_SCORE, NONE.')
parser.add_argument('--input_training', type=str, required=True, help='input file with the training data, in pickle format.')
parser.add_argument('--output_model', type=str, required=True, help='output file to receive the model, in pickle format.')
parser.add_argument('--no_group', default=False, action='store_true', help='do not group samples using id when cross-validating.')
FLAGS = parser.parse_args()

valid_svm_methods = [ 'RBF', 'LINEAR_DUAL', 'LINEAR_PRIMAL' ]
if not FLAGS.svm_method in valid_svm_methods :
    print('--svm_method must be one of ', ', '.join(valid_svm_methods), file=sys.stderr)
    sys.exit(1)
SVM_LINEAR = FLAGS.svm_method == 'LINEAR_DUAL' or FLAGS.svm_method == 'LINEAR_PRIMAL'
SVM_DUAL = FLAGS.svm_method == 'LINEAR_DUAL'

SVM_MAX_ITER = FLAGS.max_iter_svm
HYPER_MAX_ITER = FLAGS.max_iter_hyper
HYPER_JOBS = FLAGS.jobs

valid_preprocesses = [ 'PCA', 'PCA_WHITEN', 'Z_SCORE', 'NONE' ]
if not FLAGS.preprocess in valid_preprocesses :
    print('--preprocess must be one of ', ' '.join(valid_preprocesses), file=sys.stderr)
    sys.exit(1)

first = start = su.print_and_time('Reading training data...', file=sys.stderr)
ids, labels, features = su.read_pickled_data(FLAGS.input_training)
start = su.print_and_time('', past=start, file=sys.stderr)


num_samples = len(ids)
min_gamma   = np.floor(np.log2(1.0/num_samples)) - 4.0
max_gamma   = min(3.0, min_gamma+32.0)
scale_gamma = max_gamma-min_gamma
print('\tSamples: ', num_samples, file=sys.stderr)
if not SVM_LINEAR :
    print('\tGamma: ', min_gamma, min_gamma+scale_gamma, file=sys.stderr)

start = su.print_and_time('Training preprocessor...', file=sys.stderr)

if FLAGS.preprocess == 'PCA' :
    preprocessor = sk.decomposition.PCA(copy=False, whiten=False)
elif FLAGS.preprocess == 'PCA_WHITEN' :
    preprocessor = sk.decomposition.PCA(copy=False, whiten=True)
elif FLAGS.preprocess == 'Z_SCORE' :
    preprocessor = sk.preprocessing.StandardScaler(copy=False)
elif FLAGS.preprocess == 'NONE' :
    # func=None implies identity function
    preprocessor = sk.preprocessing.FunctionTransformer(func=None, inverse_func=None, validate=False,
        accept_sparse=False, pass_y=False, kw_args=None, inv_kw_args=None)
else :
    assert False, '(bug) Invalid value for FLAGS.preprocess: %s' % FLAGS.preprocess
features = preprocessor.fit_transform(features)

group_msg = 'ungrouped' if FLAGS.no_group else 'grouped'

start = su.print_and_time('====================\nTraining melanoma classifier (%s)...\n' % group_msg, past=start, file=sys.stderr)
classifier, tuning = su.new_classifier(linear=SVM_LINEAR, dual=SVM_DUAL, max_iter=SVM_MAX_ITER, min_gamma=min_gamma, scale_gamma=scale_gamma)
classifier_m = su.hyperoptimizer(classifier, tuning, max_iter=HYPER_MAX_ITER, n_jobs=HYPER_JOBS, group=not FLAGS.no_group)
classifier_m.fit(features, (labels==1).astype(np.int), groups=None if FLAGS.no_group else ids)
print('Best params:', classifier_m.best_params_, file=sys.stderr)
print('...', classifier_m.best_params_, end='', file=sys.stderr)

start = su.print_and_time('====================\nTraining keratosis classifier (%s)...\n' % group_msg, past=start, file=sys.stderr)
classifier, tuning = su.new_classifier(linear=SVM_LINEAR, dual=SVM_DUAL, max_iter=SVM_MAX_ITER, min_gamma=min_gamma, scale_gamma=scale_gamma)
classifier_k = su.hyperoptimizer(classifier, tuning, max_iter=HYPER_MAX_ITER, n_jobs=HYPER_JOBS, group=not FLAGS.no_group)
classifier_k.fit(features, (labels==2).astype(np.int), groups=None if FLAGS.no_group else ids)
print('Best params:', classifier_k.best_params_, file=sys.stderr)
print('...', classifier_k.best_params_, end='', file=sys.stderr)

start = su.print_and_time('====================\nWriting model...', past=start, file=sys.stderr)
model_file = open(FLAGS.output_model, 'wb')
pickle.dump(preprocessor, model_file)
pickle.dump(classifier_m, model_file)
pickle.dump(classifier_k, model_file)
pickle.dump(FLAGS, model_file)
model_file.close()

print('\n Total time ', end='', file=sys.stderr)
_ = su.print_and_time('Done!\n', past=first, file=sys.stderr)
