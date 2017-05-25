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
import itertools
import operator
import sys

def count_inversions(list1_images, list2_images_truths) :
    order1 = { image : order for order,image in enumerate(list1_images) }
    order2 = [ [ order1[image[0]], image[1] ] for image in list2_images_truths ]
    inversions  = sum(( 1 for i,j in itertools.combinations(xrange(len(order2)), 2) if order2[i][0]>order2[j][0] ))
    significant = sum(( 1 for i,j in itertools.combinations(xrange(len(order2)), 2) if order2[i][0]>order2[j][0] and order2[i][1]!=order2[j][1]))
    return inversions, significant

ground_truth = [ line.strip().split(',') for line in open(sys.argv[1], 'rt') ]
ground_truth = { line[0].strip() : (int(float(line[1])), int(float(line[2]))) for line in ground_truth[1:] }

submission1  = [ line.strip().split(',') for line in open(sys.argv[2], 'rt') ]
submission2  = [ line.strip().split(',') for line in open(sys.argv[3], 'rt') ]

submission1  = [ [ line[0].strip(), float(line[1]), float(line[2]), ground_truth[line[0].strip()] ] for line in submission1 ]
submission2  = [ [ line[0].strip(), float(line[1]), float(line[2]), ground_truth[line[0].strip()] ] for line in submission2 ]

# Melanoma
melanoma1 = [  s[0]           for s in sorted(submission1, key=operator.itemgetter(1)) ]
melanoma2 = [ [s[0], s[3][0]] for s in sorted(submission2, key=operator.itemgetter(1)) ]

print('Melanoma submission, test x reference inversions: %d (significant: %d)' % count_inversions(melanoma1, melanoma2))

# Keratosis
keratosis1 = [  s[0]           for s in sorted(submission1, key=operator.itemgetter(2)) ]
keratosis2 = [ [s[0], s[3][1]] for s in sorted(submission2, key=operator.itemgetter(2)) ]

print('Keratosis submission, test x reference inversions: %d (significant: %d)' % count_inversions(keratosis1, keratosis2))
