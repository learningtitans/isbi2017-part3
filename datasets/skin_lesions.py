# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Contributed 2017 Eduardo Valle. eduardovalle.com/ github.com/learningtitans
# flowers.py => skin_lesions.py
"""Provides data for the skin_lesion dataset.

The dataset scripts used to create the dataset can be found at:
prepare_skin_lesions_train.py
prepare_skin_lesions_test.py
convert_skin_lesions_dataset.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import tensorflow as tf

slim = tf.contrib.slim

_FILE_PATTERN = 'skin_lesions_%s_*.tfrecord'

_ITEMS_TO_DESCRIPTIONS = {
    'id'       : 'id of the image.',
    'case'     : 'id of the (medical) case. When present, used to avoid cross-contamination.',
    'image'    : 'A color image of varying size.',
    'label'    : 'General lesion label (0 - other, 1 - melanoma, 2 - seborrheic keratosis).',
    'melanoma' : 'Label for melanoma task (1 if melanoma).',
    'keratosis': 'Label for seborrheic keratosis task (1 if keratosis).',
    'type'     : 'Type of image (1 if dermoscopic, 0 if clinical).',
    'age'      : 'Age in years.',
    'has_age'  : 'Is age present (1 if present, 0 if missing).',
    'sex'      : 'Sex (1 if female, 0 if male).',
    'has_sex'  : 'Is sex present (1 if present, 0 if missing).',
    'schedule' : 'Training schedule for curriculum learning: lower values should go first.',
    'weight'   : 'Sample weight for weighted learning: greater values have greater label confidence.',
}


def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
  """Gets a dataset tuple with instructions for reading flowers.

  Args:
    split_name: A train/validation split name.
    dataset_dir: The base directory of the dataset sources.
    file_pattern: The file pattern to use when matching the dataset sources.
      It is assumed that the pattern contains a '%s' string so that the split
      name can be inserted.
    reader: The TensorFlow reader type.

  Returns:
    A `Dataset` namedtuple.

  Raises:
    ValueError: if `split_name` is not a valid train/validation split.
  """
  print('===================================')
  print('SPLIT', split_name)  
  print('===================================')
  SPLITS_TO_SIZES  = pickle.load(open(os.path.join(dataset_dir, 'splits_to_sizes.pkl'),  'r'))
  CLASSES_TO_SIZES = pickle.load(open(os.path.join(dataset_dir, 'classes_to_sizes.pkl'), 'r'))

  if split_name not in SPLITS_TO_SIZES:
    raise ValueError('split name %s was not recognized.' % split_name)

  if not file_pattern:
    file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

  # Allowing None in the signature so that dataset_factory can use the default.
  if reader is None:
    reader = tf.TFRecordReader

  keys_to_features = {
      'image/encoded'         : tf.FixedLenFeature((), tf.string,  default_value=''),
      'image/format'          : tf.FixedLenFeature((), tf.string,  default_value='jpg'),
      'image/class/label'     : tf.FixedLenFeature([], tf.int64,   default_value=tf.zeros([], dtype=tf.int64)),
      'image/class/melanoma'  : tf.FixedLenFeature([], tf.int64,   default_value=tf.zeros([], dtype=tf.int64)),
      'image/class/keratosis' : tf.FixedLenFeature([], tf.int64,   default_value=tf.zeros([], dtype=tf.int64)),
      'image/meta/id'         : tf.FixedLenFeature([], tf.string,  default_value=''),
      'image/meta/case'       : tf.FixedLenFeature((), tf.string,  default_value=''),
      'image/meta/type'       : tf.FixedLenFeature((), tf.float32, default_value=tf.zeros([], dtype=tf.float32)),
      'image/meta/age'        : tf.FixedLenFeature([], tf.float32, default_value=tf.zeros([], dtype=tf.float32)),
      'image/meta/has_age'    : tf.FixedLenFeature([], tf.float32, default_value=tf.zeros([], dtype=tf.float32)),
      'image/meta/sex'        : tf.FixedLenFeature([], tf.float32, default_value=tf.zeros([], dtype=tf.float32)),
      'image/meta/has_sex'    : tf.FixedLenFeature([], tf.float32, default_value=tf.zeros([], dtype=tf.float32)),
      'image/meta/schedule'   : tf.FixedLenFeature([], tf.float32, default_value=tf.zeros([], dtype=tf.float32)),
      'image/meta/weight'     : tf.FixedLenFeature([], tf.float32, default_value=tf.zeros([], dtype=tf.float32)),
  }

  items_to_handlers = {
      'image'    : slim.tfexample_decoder.Image(),
      'label'    : slim.tfexample_decoder.Tensor('image/class/label'),
      'melanoma' : slim.tfexample_decoder.Tensor('image/class/melanoma'),
      'keratosis': slim.tfexample_decoder.Tensor('image/class/keratosis'),
      'id'       : slim.tfexample_decoder.Tensor('image/meta/id'),      
      'case'     : slim.tfexample_decoder.Tensor('image/meta/case'),      
      'type'     : slim.tfexample_decoder.Tensor('image/meta/type'),      
      'age'      : slim.tfexample_decoder.Tensor('image/meta/age'),       
      'has_age'  : slim.tfexample_decoder.Tensor('image/meta/has_age'),   
      'sex'      : slim.tfexample_decoder.Tensor('image/meta/sex'),       
      'has_sex'  : slim.tfexample_decoder.Tensor('image/meta/has_sex'),   
      'schedule' : slim.tfexample_decoder.Tensor('image/meta/schedule'),  
      'weight'   : slim.tfexample_decoder.Tensor('image/meta/weight'),
  }

  decoder = slim.tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  labels_to_names = { 0 : 'other', 1 : 'melanoma', 2 : 'keratosis' }
  num_classes = 3

  return slim.dataset.Dataset(
      data_sources=file_pattern,
      reader=reader,
      decoder=decoder,
      num_samples=SPLITS_TO_SIZES[split_name],
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      num_classes=num_classes,
      labels_to_names=labels_to_names)
