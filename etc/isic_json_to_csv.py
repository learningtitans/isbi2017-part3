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
import glob
import json
import os.path
import sys

labels = open('isic_archive_labels.txt', 'w')

print(', '.join(['image', 'benign_malignant', 'diagnosis', 'diagnosis_confirm_type', 'age_approx', 'sex' ]), file=labels)

benign_malignant_set = set()
diagnosis_set = set()
diagnosis_confirm_type_set = set()
age_approx_set = set()
sex_set = set()
image_count = 0
benign_malignant_count = 0
diagnosis_count = 0
diagnosis_confirm_type_count = 0
age_approx_count = 0
sex_count = 0

file_count = 0
for metafile in glob.glob('meta/*.json') :
    id = os.path.basename(metafile).split('.')[0]
    imagefile = id + '.jpg'
    if not os.path.isfile(os.path.join('images', imagefile)) :
        print('\nMissing image file: ' + imagefile, file=sys.stderr)
        continue
    else :
        file_count += 1
        print('.', end='\n' if file_count%100==0 else '', file=sys.stderr)

    meta = json.loads(open(metafile, 'r').read())
    meta = meta.get(u'meta', {})
    clinical = meta.get(u'clinical', {})
    benign_malignant = clinical.get(u'benign_malignant')
    diagnosis = clinical.get(u'diagnosis')
    diagnosis_confirm_type = clinical.get(u'diagnosis_confirm_type')
    age_approx = clinical.get(u'age_approx')
    sex = clinical.get(u'sex')

    image_count += 1
    benign_malignant_count += 0 if benign_malignant is None else 1
    diagnosis_count += 0 if diagnosis is None else 1
    diagnosis_confirm_type_count += 0 if diagnosis_confirm_type is None else 1
    age_approx_count += 0 if age_approx is None else 1
    sex_count += 0 if sex is None else 1

    fields = [id, benign_malignant, diagnosis, diagnosis_confirm_type, age_approx, sex]
    fields = [ str(f) for f in fields ]
    print(', '.join(fields), file=labels)

    benign_malignant_set.add(benign_malignant)
    diagnosis_set.add(diagnosis)
    diagnosis_confirm_type_set.add(diagnosis_confirm_type)
    age_approx_set.add(age_approx)
    sex_set.add(sex)

print('', file=sys.stderr)

infos = open('isic_archive_infos.txt', 'w')
print(', '.join(['images', 'benign_malignant', 'diagnosis', 'diagnosis_confirm_type', 'age_approx', 'sex' ]), file=infos)
fields = [image_count, benign_malignant_count, diagnosis_count, diagnosis_confirm_type_count, age_approx_count, sex_count]
fields = [ str(f) for f in fields ]
print(', '.join(fields), file=infos)
print('benign_malignant_set: ' + str(benign_malignant_set), file=infos)
print('diagnosis_set: ' + str(diagnosis_set), file=infos)
print('diagnosis_confirm_type_set: ' + str(diagnosis_confirm_type_set), file=infos)
print('age_approx_set: ' + str(age_approx_set), file=infos)
print('sex_set: ' + str(sex_set), file=infos)
