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
import pickle
import os
import sys

import numpy as np
import scipy as sp


if sys.argv[1]=='LOGITS' :
    use_logits = True
    all_combinations = False
elif sys.argv[1]=='ALL_LOGITS' :
    use_logits = True
    all_combinations = True
elif sys.argv[1]=='PROBS' :
    use_logits = False
    all_combinations = False
elif sys.argv[1]=='ALL_PROBS' :
    use_logits = False
    all_combinations = True
else :
    print('ERROR: First argument must be [ALL_]LOGITS or [ALL_]PROBS, instead got %s' % sys.argv, file=sys.stderr)
    sys.exit(1)
source_folder = sys.argv[2]
target_file = sys.argv[3]
labels_file = sys.argv[4] if len(sys.argv)>4 else None

FEATURE_LENGTH = 2

feature_files = \
    {
        'rc25' :         { 1 : 'partial_rc25_1.txt',
                           2 : 'partial_rc25_2.txt',
                           3 : 'partial_rc25_3.txt', },
        'rc25_50_s' :    { 1 : 'partial_rc25_50_s_1.txt',
                           2 : 'partial_rc25_50_s_2.txt',
                           3 : 'partial_rc25_50_s_3.txt', },
        'rc25_max' :     { 1 : 'partial_rc25_max_1.txt',
                           2 : 'partial_rc25_max_2.txt',
                           3 : 'partial_rc25_max_3.txt', },
        'rc28' :         { 1 : 'partial_rc28_1.txt',
                           2 : 'partial_rc28_2.txt',
                           3 : 'partial_rc28_3.txt', },
        'rc28_50_s' :    { 1 : 'partial_rc28_50_s_1.txt',
                           2 : 'partial_rc28_50_s_2.txt',
                           3 : 'partial_rc28_50_s_3.txt', },
        'rc28_50avg_s' : { 1 : 'partial_rc28_50avg_s_1.txt',
                           2 : 'partial_rc28_50avg_s_2.txt',
                           3 : 'partial_rc28_50avg_s_3.txt', },
        'rc30' :         { 1 : 'partial_rc30_1.txt',
                           2 : 'partial_rc30_2.txt',
                           3 : 'partial_rc30_3.txt', },
    }

def bits_from_probstr(probstr) :
    prob = float(probstr)
    prob = max(bits_from_probstr.SCEPTICISM_THRESHOLD, prob)
    prob = min(1.0-bits_from_probstr.SCEPTICISM_THRESHOLD, prob)
    odds = prob/(1.0-prob)
    bits = np.log2(odds)
    return bits
# This corresponds to doubting the model can do better than one error in one million
bits_from_probstr.SCEPTICISM_THRESHOLD=0.00001

if use_logits :
    getfeat = bits_from_probstr
else :
    getfeat = float

image_ids = None
feature_sets = {}
for feature_key in feature_files.keys() :
    for replica_key in feature_files[feature_key].keys() :
        features = [ line.strip().split(',')
            for line in open(os.path.join(source_folder, feature_files[feature_key][replica_key]), 'r') ]
        if not all([ len(line)==FEATURE_LENGTH+1 for line in  features ]) :
            print('WARNING: At least one sample with wrong feature length in %s' % feature_files[feature_key][replica_key], file=sys.stderr)
        features = { f[0].strip() : [ getfeat(ff.strip()) for ff in f[1:] ] for f in features }
        feature_sets.setdefault(feature_key, {})[replica_key] = features
        feature_ids = sorted(features.keys())
        if image_ids is None :
            image_ids = feature_ids
        else :
            if feature_ids != image_ids  :
                print('WARNING: mismatched image id list in %s' % feature_files[feature_key][replica_key], file=sys.stderr)

if not labels_file is None :
    labels = [ line.strip().split(',') for line in open(labels_file, 'r') ]
    labels = { label[0].strip() : 1 if label[7].strip()=='1' else (2 if label[8].strip()=='1' else 0)
        for label in labels[1:] }
else :
    print('WARNING: no label file informed --- assuming this is a test split', file=sys.stderr)
    labels = { iid : 0 for iid in image_ids }

feature_order = tuple(('rc25', 'rc25_50_s', 'rc25_max', 'rc28', 'rc28_50_s', 'rc28_50avg_s', 'rc30' ))
if all_combinations :
    # Got with random.sample(list(itertools.product([1,2,3], repeat=7)), 97), then added [1, 1, ..., 1] etc. by hand
    feature_combinations = \
        [(1, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 2, 1), (1, 1, 1, 3, 3, 3, 3), (1, 1, 2, 1, 2, 2, 2), (1, 1, 2, 1, 2, 3, 3),
         (1, 1, 2, 1, 3, 2, 2), (1, 1, 2, 2, 3, 2, 1), (1, 1, 2, 3, 1, 1, 1), (1, 1, 3, 2, 1, 3, 3), (1, 1, 3, 2, 2, 3, 3),
         (1, 1, 3, 2, 3, 2, 1), (1, 1, 3, 3, 1, 1, 1), (1, 2, 1, 2, 2, 1, 1), (1, 2, 1, 2, 2, 1, 2), (1, 2, 1, 2, 2, 3, 2),
         (1, 2, 1, 2, 3, 1, 2), (1, 2, 2, 1, 2, 1, 3), (1, 2, 2, 1, 2, 2, 3), (1, 2, 2, 1, 2, 3, 3), (1, 2, 2, 1, 3, 3, 1),
         (1, 2, 2, 2, 1, 1, 1), (1, 2, 2, 3, 2, 3, 3), (1, 2, 3, 1, 3, 1, 2), (1, 2, 3, 2, 2, 1, 2), (1, 3, 1, 2, 1, 1, 1),
         (1, 3, 1, 3, 1, 1, 1), (1, 3, 1, 3, 2, 2, 2), (1, 3, 2, 1, 2, 1, 3), (1, 3, 2, 1, 2, 2, 3), (1, 3, 2, 1, 3, 3, 3),
         (1, 3, 2, 2, 2, 2, 3), (1, 3, 2, 3, 1, 1, 2), (1, 3, 2, 3, 1, 2, 1), (1, 3, 2, 3, 1, 3, 2), (1, 3, 3, 1, 2, 1, 1),
         (1, 3, 3, 1, 3, 2, 3), (1, 3, 3, 2, 3, 1, 3), (1, 3, 3, 3, 1, 2, 1), (2, 1, 1, 2, 2, 3, 2), (2, 1, 1, 3, 2, 3, 2),
         (2, 1, 2, 2, 1, 3, 2), (2, 1, 2, 2, 3, 2, 3), (2, 1, 2, 3, 2, 2, 2), (2, 1, 2, 3, 3, 1, 1), (2, 1, 2, 3, 3, 3, 1),
         (2, 1, 3, 3, 3, 2, 2), (2, 1, 3, 3, 3, 3, 3), (2, 2, 1, 1, 1, 3, 3), (2, 2, 1, 2, 1, 3, 3), (2, 2, 2, 1, 3, 2, 2),
         (2, 2, 2, 2, 2, 1, 1), (2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 3, 3, 1), (2, 2, 3, 1, 2, 1, 3), (2, 2, 3, 1, 2, 2, 3),
         (2, 2, 3, 3, 1, 1, 2), (2, 2, 3, 3, 2, 2, 2), (2, 3, 1, 3, 1, 3, 1), (2, 3, 1, 3, 3, 2, 3), (2, 3, 2, 2, 3, 2, 2),
         (2, 3, 2, 3, 1, 3, 1), (2, 3, 3, 1, 1, 2, 1), (2, 3, 3, 1, 3, 2, 1), (2, 3, 3, 2, 2, 2, 1), (2, 3, 3, 2, 2, 3, 2),
         (2, 3, 3, 3, 1, 2, 3), (3, 1, 1, 1, 2, 1, 1), (3, 1, 1, 1, 2, 2, 3), (3, 1, 1, 2, 1, 2, 1), (3, 1, 1, 3, 2, 1, 3),
         (3, 1, 2, 1, 1, 1, 3), (3, 1, 2, 1, 2, 3, 1), (3, 1, 2, 1, 3, 2, 3), (3, 1, 2, 2, 3, 1, 2), (3, 1, 2, 3, 3, 1, 2),
         (3, 1, 3, 1, 1, 2, 2), (3, 1, 3, 2, 1, 1, 2), (3, 1, 3, 2, 3, 1, 3), (3, 1, 3, 2, 3, 3, 2), (3, 1, 3, 3, 1, 1, 2),
         (3, 2, 1, 1, 1, 1, 1), (3, 2, 1, 1, 2, 1, 2), (3, 2, 1, 2, 3, 1, 1), (3, 2, 1, 3, 2, 1, 2), (3, 2, 2, 1, 3, 1, 2),
         (3, 2, 2, 2, 1, 3, 2), (3, 2, 2, 3, 2, 1, 1), (3, 2, 3, 1, 1, 3, 1), (3, 2, 3, 1, 3, 1, 1), (3, 2, 3, 3, 1, 3, 1),
         (3, 3, 1, 1, 1, 3, 1), (3, 3, 1, 1, 2, 3, 1), (3, 3, 1, 2, 3, 2, 3), (3, 3, 1, 2, 3, 3, 3), (3, 3, 2, 1, 2, 1, 1),
         (3, 3, 2, 1, 2, 2, 2), (3, 3, 2, 2, 2, 1, 3), (3, 3, 3, 2, 1, 1, 2), (3, 3, 3, 2, 3, 3, 2), (3, 3, 3, 3, 3, 3, 3)]
else :
    feature_combinations = [ [1]*7, [2]*7, [3]*7 ]

num_samples = len(image_ids)*len(feature_combinations)

target_file = open(target_file, 'wb')
pickle.dump([ num_samples, len(feature_order)*FEATURE_LENGTH ], target_file)
for iid in image_ids :
    for comb in feature_combinations :
        feature = []
        for f, fname in enumerate(feature_order) :
            feature.extend(feature_sets[fname][comb[f]][iid])
        pickle.dump([ iid, labels[iid], feature ], target_file)

