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

parser = argparse.ArgumentParser(prog='train_svm_layer.py', description='Predict the SVM decision.')
parser.add_argument('--input_model', type=str, required=True, help='input trained model, in pickle format.')
parser.add_argument('--input_test', type=str, required=True, help='input file with the test data, in pickle format.')
parser.add_argument('--output_predictions', type=str , help='input file with the test data, in isbi challenge format (default=stdout).')
parser.add_argument('--output_metrics', type=str, help='input file with the test data, in text format (default=stdout).')
parser.add_argument('--pool_by_id', type=str, default='none', help='pool answers of contiguous identical ids: none (default), avg, max, xtrm')
FLAGS = parser.parse_args()

first = start = su.print_and_time('Reading trained model...', file=sys.stderr)
model_file = open(FLAGS.input_model, 'rb')
preprocessor = pickle.load(model_file)
classifier_m = pickle.load(model_file)
classifier_k = pickle.load(model_file)
model_file.close()

start = su.print_and_time('Reading test data...',  past=start, file=sys.stderr)
image_ids, labels, features = su.read_pickled_data(FLAGS.input_test)
num_samples = len(image_ids)

start = su.print_and_time('Preprocessing test data...', file=sys.stderr)
features = preprocessor.transform(features)

# "Probabilities" should come between quotes here
# Only if the scores are true logits the probabilities will be consistent
def probability_from_logits(logits) :
    odds = np.exp(logits)
    return odds/(odds+1.0)
def logits_from_probability(prob) :
    with np.errstate(divide='ignore') :
      odds = prob/(1.0-prob)
      return np.log(odds)
def extreme_probability(prob) :
  return prob[np.argmax(np.abs(logits_from_probability(prob)))]

start = su.print_and_time('Predicting test data...\n', past=start, file=sys.stderr)
predictions_m = probability_from_logits(classifier_m.decision_function(features))
predictions_k = probability_from_logits(classifier_k.decision_function(features))


outfile = open(FLAGS.output_predictions, 'w') if FLAGS.output_predictions else sys.stdout
if FLAGS.pool_by_id=='none' :
  for i in xrange(len(image_ids)) :
    print(image_ids[i], predictions_m[i], predictions_k[i], sep=',', file=outfile)
else :
  previous_id = None
  def print_result() :
    if FLAGS.pool_by_id=='avg' :
      print(previous_id, np.mean(all_m), np.mean(all_k), sep=',', file=outfile)
    elif FLAGS.pool_by_id=='max' :
      print(previous_id, np.amax(all_m), np.amax(all_k), sep=',', file=outfile)
    elif FLAGS.pool_by_id=='xtrm' :
      print(previous_id, extreme_probability(all_m), extreme_probability(all_k), sep=',', file=outfile)
    else :
      raise ValueError('Invalid value for FLAGS.pool_by_id: %s' % FLAGS.pool_by_id)

  for i in xrange(len(image_ids)) :
    if image_ids[i]!=previous_id :
      if previous_id is not None :
        print_result()
      previous_id = image_ids[i]
      all_m = np.asarray([ predictions_m[i] ])
      all_k = np.asarray([ predictions_k[i] ])
    else :
      all_m = np.concatenate((all_m, np.asarray([ predictions_m[i] ])))
      all_k = np.concatenate((all_k, np.asarray([ predictions_k[i] ])))
  if previous_id is not None :
    print_result()


metfile = open(FLAGS.output_metrics, 'w') if FLAGS.output_metrics else sys.stderr
try :
  accs = []
  aucs = []
  mAPs = []
  for j, scores_j in [ [1, predictions_m], [2, predictions_k] ] :
    labels_j = (labels == j).astype(np.int)
    acc = sk.metrics.accuracy_score(labels, scores_j.astype(np.int))
    print('Acc: ', acc, file=metfile)
    accs.append(acc)
    auc = sk.metrics.roc_auc_score(labels_j, scores_j)
    aucs.append(auc)
    print('AUC[%d]: ' % j, auc, file=metfile)
    mAP = sk.metrics.average_precision_score(labels_j, scores_j)
    mAPs.append(mAP)
    print('mAP[%d]: ' % j, mAP, file=metfile)
  print('Acc_avg: ', sum(accs) / 2.0, file=metfile)
  print('AUC_avg: ', sum(aucs) / 2.0, file=metfile)
  print('mAP_avg: ', sum(mAPs) / 2.0, file=metfile)
except ValueError :
  pass

print('\n Total time ', end='', file=sys.stderr)
_ = su.print_and_time('Done!\n', past=first, file=sys.stderr)
