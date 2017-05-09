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
import os
import sys

if len(sys.argv)<=2 or (not sys.argv[2] in ['ALL', 'CHALLENGE_2017', 'DEPLOY_2017']) :
    print('usage: prepare_skin_lesions_train.py images_path < ALL | CHALLENGE_<YEAR> | DEPLOY_<YEAR> > suspected_duplicate_cases.msg')
    sys.exit(1)

images_path = sys.argv[1]
prepare_challenge = (sys.argv[2] == 'CHALLENGE_2017') or (sys.argv[2] == 'DEPLOY_2017')
final_deployment = (sys.argv[2] == 'DEPLOY_2017')

def checkImage(image, previous) :
    if image == previous :
        print('duplicate ' + image + ' --- skipping', file=sys.stderr)
        return False
    if not os.path.isfile(os.path.join(images_path, image) + '.jpg') :
        print('missing ' + image + '.jpg --- skipping', file=sys.stderr)
        return False
    return True

# ==== Header ====
#         0     1     2    3      4      5      6         7          8          9       10      11
print('image, case, type, age, has_age, sex, has_sex, melanoma, keratosis, schedule, weight, dataset')


# ==== EDRA Atlas of Dermoscopy ====

firstLine = True
previous = ''
for rec in open('atlas.csv', 'r') :
    if firstLine : firstLine = False; continue
    f = rec.strip().split('\t')
    image     = f[1]
    case      = f[2].upper()
    imgtype   = f[0][0].lower()
    age       = f[6] if f[7]=='1' else ''
    has_age   = f[7]
    sex       = f[8] if f[9]=='1' else ''
    has_sex   = f[9]
    melanoma  = f[4]
    keratosis = f[5]
    schedule  = f[10]
    weight    = f[11]
    dataset   = 'atlas'
    if not checkImage(image, previous) :
        continue
    print(image, case, imgtype, age, has_age, sex, has_sex, melanoma, keratosis, schedule, weight, dataset, sep=', ')
    previous = image


# ==== ISIC Archive (full) ====

if final_deployment :
    near_duplicates = [ line.strip().split(',') for line in open(sys.argv[3], 'r') ]
    near_duplicates = { case[0].strip() : case[1].strip() for case in near_duplicates }

weight_map = { 'histopathology' : 3,
               'single image expert consensus' : 2 }
firstLine = True
for rec in open('isic.txt', 'r') :
    if firstLine : firstLine = False; continue
    f = [ f.strip() for f in rec.strip().split(',') ]

    if (prepare_challenge and
        (f[2] == 'atypical melanocytic proliferation' or
         f[2] == 'other' or
         f[2] == 'None' or
         f[4] == '15')) :
        # Skips uncertain labels that might confound classification
        # Skips all images with age == 15 due to bias problems
        continue

    image     = f[0]
    case      = near_duplicates.get(image, '') if final_deployment else ''
    imgtype   = 'd'
    age       = f[4] if f[4]!='None' else ''
    has_age   = '1'  if f[4]!='None' else '0'
    sex       = f[5] if f[5]!='None' else ''
    has_sex   = '1'  if f[5]!='None' else '0'
    melanoma  = '1' if f[2]=='melanoma'             else '0'
    keratosis = '1' if f[2]=='seborrheic keratosis' else '0'
    schedule  = '2'
    weight    = weight_map.get(f[3], 1)
    dataset   = 'isic'
    if not checkImage(image, previous) :
        continue
    print(image, case, imgtype, age, has_age, sex, has_sex, melanoma, keratosis, schedule, weight, dataset, sep=', ')
    previous = image


# ==== Official ISIC Challenge Data ====

firstLine = True
for rec in itertools.izip(
        open('ISIC-2017_Training_Data_metadata.csv', 'r'),
        open('ISIC-2017_Training_Part3_GroundTruth.csv', 'r') ) :
    if firstLine : firstLine = False; continue
    f = [ f.strip() for f in rec[0].strip().split(',')+rec[1].strip().split(',') ]
    if f[0] != f[3] :
        print('Mismatched record between metadata and ground truth: ' + f[0] +
            ' vs. ' + f[3], file=sys.stderr)
        continue

    image     = f[0]
    case      = near_duplicates.get(image, '') if final_deployment else ''
    imgtype   = 'd'
    age       = f[1] if f[1]!='unknown' else ''
    has_age   = '1'  if f[1]!='unknown' else '0'
    sex       = f[2] if f[2]!='unknown' else ''
    has_sex   = '1'  if f[2]!='unknown' else '0'
    melanoma  = int(float(f[4]))
    keratosis = int(float(f[5]))
    schedule  = '2'
    weight    = '5'
    dataset   = 'challenge'
    if not checkImage(image, previous) :
        continue
    print(image, case, imgtype, age, has_age, sex, has_sex, melanoma, keratosis, schedule, weight, dataset, sep=', ')
    previous = image


# ==== IRMA ====

firstLine = True
for rec in open('irma.csv', 'r') :
    if firstLine : firstLine = False; continue
    f = [ f.strip() for f in rec.strip().split(',') ]

    if (prepare_challenge and f[1]!='1') :
        # Skips non-melanomas, because they might be keratosis
        continue

    image     = f[0]
    case      = near_duplicates.get(image, '') if final_deployment else ''
    imgtype   = 'd'
    age       = ''
    has_age   = '0'
    sex       = ''
    has_sex   = '0'
    melanoma  = '1'
    keratosis = '0'
    schedule  = '1'
    weight    = '1'
    dataset   = 'irma'
    if not checkImage(image, previous) :
        continue
    print(image, case, imgtype, age, has_age, sex, has_sex, melanoma, keratosis, schedule, weight, dataset, sep=', ')
    previous = image


# ==== PH2 ====

firstLine = True
for rec in open('ph2.csv', 'r') :
    if firstLine : firstLine = False; continue
    f = [ f.strip() for f in rec.strip().split(',') ]

    if (prepare_challenge and len(f[3])>0) :
        # Skips atypical nevi, because they might be keratosis
        continue

    image     = f[0]
    case      = near_duplicates.get(image, '') if final_deployment else ''
    imgtype   = 'd'
    age       = ''
    has_age   = '0'
    sex       = ''
    has_sex   = '0'
    melanoma  = '1' if len(f[4])>0 else '0'
    keratosis = '0'
    schedule  = '1'
    weight    = '1' if len(f[1])==0 else '3' # Histopatological confirmation
    dataset   = 'ph2'
    if not checkImage(image, previous) :
        continue
    print(image, case, imgtype, age, has_age, sex, has_sex, melanoma, keratosis, schedule, weight, dataset, sep=', ')
    previous = image


# ==== Edinburgh  ====

firstLine = True
for rec in open('dermofit.csv', 'r') :
    if firstLine : firstLine = False; continue
    f = [ f.strip() for f in rec.strip().split(',') ]

    image     = f[0]
    case      = near_duplicates.get(image, '') if final_deployment else ''
    imgtype   = 'd'
    age       = ''
    has_age   = '0'
    sex       = ''
    has_sex   = '0'
    melanoma  = '1' if f[1]=='MEL' else '0'
    keratosis = '1' if f[1]=='SK'  else '0'
    schedule  = '1'
    weight    = '1'
    dataset   = 'dermofit'
    if not checkImage(image, previous) :
        continue
    print(image, case, imgtype, age, has_age, sex, has_sex, melanoma, keratosis, schedule, weight, dataset, sep=', ')
    previous = image
