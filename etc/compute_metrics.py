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
from __future__ import print_function
import sys

import numpy as np
import sklearn.metrics


truth = [ line.strip().split(',') for line in open(sys.argv[1], 'rt') ]
submission = [ line.strip().split(',') for line in open(sys.argv[2], 'rt') ]

submission = [ [ line[0].strip(), float(line[1]), float(line[2]) ] for line in submission ]
truth = [ [ line[0].strip(), float(line[1]), float(line[2]) ] for line in truth[1:] ]

# Melanoma
np_scores = np.asarray([ s[1] for s in submission ])
np_labels = np.asarray([ t[1] for t in truth ])
m_auc = sklearn.metrics.roc_auc_score(np_labels, np_scores)
print('Melanoma AUC: %f' % m_auc)

# Keratosis
np_scores = np.asarray([ s[2] for s in submission ])
np_labels = np.asarray([ t[2] for t in truth ])
k_auc = sklearn.metrics.roc_auc_score(np_labels, np_scores)
print('Keratosis AUC: %f' % k_auc)

# Average
print('Average AUC: %f' % ((m_auc+k_auc)/2.0))
