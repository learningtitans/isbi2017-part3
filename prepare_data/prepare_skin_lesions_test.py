# Copyright 2017 Eduardo Valle. All Rights Reserved.
# eduardovalle.com/  github.com/learningtitans
#
# Licensed under 
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import print_function
import itertools
import sys

print('image, (case), type, age, has_age, sex, has_sex, (melanoma), (keratosis), (schedule), (weight)')

firstLine = True
previous = ''
for rec in open(sys.argv[1], 'r') :
    if firstLine : firstLine = False; continue
    f = [ f.strip() for f in rec.strip().split(',') ]

    image     = f[0]
    case      = ''
    imgtype   = 'd'
    age       = f[1] if f[1]!='unknown' else ''
    has_age   = '1'  if f[1]!='unknown' else '0'
    sex       = f[2] if f[2]!='unknown' else ''
    has_sex   = '1'  if f[2]!='unknown' else '0'
    melanoma  = 0
    keratosis = 0
    schedule  = 0
    weight    = 0
    if image == previous :
        print('duplicate ' + image, file=sys.stderr)
    print(image, case, imgtype, age, has_age, sex, has_sex, melanoma, keratosis, schedule, weight, sep=', ')
    previous = image
