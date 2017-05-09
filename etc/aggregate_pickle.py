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
import pickle
import numpy as np
import sys

source = open(sys.argv[1], 'rb')
target = open(sys.argv[2], 'wb')

header = pickle.load(source)
num_samples = header[0]
feature_size = header[1]

if num_samples%50!=0 :
    print("Number of records not a multiple of 50")
    sys.exit(1)

num_samples = num_samples/50
header[0] = num_samples
pickle.dump(header, target)

for s in xrange(num_samples) :
    print(s, '              ', end='\r')
    first = pickle.load(source)
    for r in xrange(49) :
        rec = pickle.load(source)
        if rec[0]!=first[0] :
            print("Aggregations requires groups of 50 contiguous records")
            sys.exit(1)
        first[2] += rec[2] # Add features
    # Gets average feature
    first[2] = first[2] / 50.0
    pickle.dump(first, target)

print('\nDone!')
