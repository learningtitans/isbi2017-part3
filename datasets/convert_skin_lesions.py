# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# Contributed 2017 Eduardo Valle. eduardovalle.com/ github.com/learningtitans
# download_and_convert_flowers.py => convert_skin_lesions.py
r"""Converts Melanoma data to TFRecords of TF-Example protos.

This reads the files that make up the Melanoma data and creates three
TFRecord datasets: train, validation, and test. Each TFRecord dataset
is comprised of a set of TF-Example protocol buffers, each of which contain
a single image and label.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import math
from operator import itemgetter
import os
import pickle
import random
import sys

import tensorflow as tf


# Copied from deleted dataset_utils.py ===>
def int64_feature(values):
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def float_feature(values):
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))

def bytes_feature(values):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))
# <===


# Seed for repeatability.
_RANDOM_SEED = 0

# The number of elements per shard
_NUM_PER_SHARD = 1024

# Fractions for training, validation, and testing split
_TRAINING_PERC   = 85
_VALIDATION_PERC = 15
_TESTING_PERC    = 00

assert (_TRAINING_PERC + _VALIDATION_PERC + _TESTING_PERC) == 100

class ImageReader(object) :
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self) :
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data) :
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data) :
    image = sess.run(self._decode_jpeg,
      feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _get_dataset_filename(dataset_dir, split_name, shard_id, num_shards):
  output_filename = 'skin_lesions_%s_%05d-of-%05d.tfrecord' % (
    split_name, shard_id+1, num_shards)
  return os.path.join(dataset_dir, output_filename)


def _image_to_tfexample(image_data, image_format, height, width, metadata) :
  return tf.train.Example(features=tf.train.Features(feature={
    'image/encoded'         : bytes_feature(image_data),
    'image/format'          : bytes_feature(image_format),
    'image/class/label'     : int64_feature(1 if metadata[7]=='1' else (2 if metadata[8]=='1' else 0)),
    'image/class/melanoma'  : int64_feature(int(metadata[7])),
    'image/class/keratosis' : int64_feature(int(metadata[8])),
    'image/height'          : int64_feature(height),
    'image/width'           : int64_feature(width),
    'image/meta/id'         : bytes_feature(metadata[0]),
    'image/meta/case'       : bytes_feature(metadata[1]),
    'image/meta/type'       : float_feature(1.0 if metadata[2]=='d' else 0.0),
    'image/meta/age'        : float_feature(float(metadata[3]) if len(metadata[3])>0 else 0.0),
    'image/meta/has_age'    : float_feature(float(metadata[4])),
    'image/meta/sex'        : float_feature(1.0 if metadata[5]=='female' else 0.0),
    'image/meta/has_sex'    : float_feature(float(metadata[6])),
    'image/meta/schedule'   : float_feature(float(metadata[9])),
    'image/meta/weight'     : float_feature(float(metadata[10])),
  }))


def _convert_dataset(split_name, metadata, images_dir, dataset_dir):
  """Converts the given images and metadata to a TFRecord dataset.

  Args:
    split_name: The name of the dataset: 'train', 'validation', or 'test'
    metadata: A list with the dataset metadata
    images_dir: The directory with the input .jpg images
    dataset_dir: The directory where the converted datasets are stored.
  """
  assert split_name in ['train', 'test', 'validation']

  dataset_size = len(metadata)
  metadata = iter(metadata)
  num_shards = int(math.ceil(dataset_size / _NUM_PER_SHARD))
  if ( dataset_size % _NUM_PER_SHARD < int(_NUM_PER_SHARD/3.0) ) :
    num_shards = max(num_shards-1, 1)

  with tf.Graph().as_default(), tf.Session('') as session :
    image_reader = ImageReader()

    for shard_id in range(num_shards) :
      output_filename = _get_dataset_filename(dataset_dir, split_name, shard_id, num_shards)
      tfrecord_writer = tf.python_io.TFRecordWriter(output_filename);

      start_ndx = shard_id*_NUM_PER_SHARD
      end_ndx   = (shard_id+1)*_NUM_PER_SHARD if shard_id<num_shards-1 else dataset_size
      for i in range(start_ndx, end_ndx) :
        sys.stdout.write('\r>> Converting image %d/%d shard %d' %
          (i+1, dataset_size, shard_id))
        sys.stdout.flush()

        # Read the filename:
        meta = metadata.next()
        image_file = os.path.join(images_dir, meta[0]) + '.jpg'
        image_data = tf.gfile.FastGFile(image_file, 'r').read()
        height, width = image_reader.read_image_dims(session, image_data)

        example = _image_to_tfexample(image_data, 'jpg', height, width, meta)
        tfrecord_writer.write(example.SerializeToString())

      tfrecord_writer.close()
  sys.stdout.write('\n')
  sys.stdout.flush()


def run(mode, metadata_file, images_dir, dataset_dir, blacklist_file=None) :
  """Runs the download and conversion operation.

  Args:
    mode: TRAIN or TEST
    metadata_file: The name of the file with the dataset metadata
    images_dir: The directory with the input .jpg images
    dataset_dir: The dataset directory where the dataset is stored
  """
  if not tf.gfile.Exists(dataset_dir):
    tf.gfile.MakeDirs(dataset_dir)


  # Get metadata
  metadata = [ m.strip().split(',') for m in open(metadata_file, 'r').readlines() ]
  metadata = [ [ f.strip() for f in m] for m in metadata ]
  metadata = metadata[1:] # Skips header

  # Get blacklist
  if not blacklist_file is None :
    blacklist = [ r.strip() for r in open(blacklist_file, 'r') ]
    metadata  = [ m for m in metadata if not (m[0] in blacklist) ]

  dataset_size = len(metadata)


  if mode == 'TRAIN' :
    # Divide metadata into stratified sets
    meta_melanoma  = [ m for m in metadata if m[7]=='1' ]
    meta_keratosis = [ m for m in metadata if m[8]=='1' ]
    meta_other     = [ m for m in metadata if m[7]!='1' and m[8]!='1' ]

    CLASSES_TO_SIZES = { 'melanoma'  : len(meta_melanoma),
                         'keratosis' : len(meta_keratosis),
                         'other'     : len(meta_other) }

    # Preprocess cases division: an annoying detail is that cases
    # cannot be broken (subtle contamination)

    # ...extract all non-empty cases
    cases_melanoma  = [ m for m in meta_melanoma  if len(m[1])>0 ]
    cases_keratosis = [ m for m in meta_keratosis if len(m[1])>0 ]
    cases_other     = [ m for m in meta_other     if len(m[1])>0 ]
    meta_melanoma   = [ m for m in meta_melanoma  if len(m[1])==0 ]
    meta_keratosis  = [ m for m in meta_keratosis if len(m[1])==0 ]
    meta_other      = [ m for m in meta_other     if len(m[1])==0 ]

    # ...if there are duplicate cases, keep only one in each set
    def split_cases(cases) :
        cases.sort(key=itemgetter(1))
        first_cases = []
        next_cases  = {}
        seen_cases  = set()
        for m in cases :
          assert(len(m[1].strip())>0)
          if m[1] in seen_cases :
            next_cases[m[1]].append(m)
          else :
            seen_cases.add(m[1])
            first_cases.append(m)
            next_cases[m[1]] = []
        return first_cases, next_cases

    cases_melanoma,  extra_melanoma  = split_cases(cases_melanoma)
    cases_keratosis, extra_keratosis = split_cases(cases_keratosis)
    cases_other,     extra_other     = split_cases(cases_other)

    meta_melanoma.extend(cases_melanoma)
    meta_keratosis.extend(cases_keratosis)
    meta_other.extend(cases_other)

    # Divide into train, validation, and test sets...
    random.seed(_RANDOM_SEED)
    random.shuffle(meta_melanoma)
    random.shuffle(meta_keratosis)
    random.shuffle(meta_other)

    # ...melanomas
    n_melanoma   = len(meta_melanoma)
    n_training   = int(n_melanoma/100.0*_TRAINING_PERC)
    n_validation = int(n_melanoma/100.0*_VALIDATION_PERC)
    training     = meta_melanoma[:n_training]
    validation   = meta_melanoma[n_training:n_training+n_validation]
    test         = meta_melanoma[n_training+n_validation:]
    # ...keratosis
    n_keratosis  = len(meta_keratosis)
    n_training   = int(n_keratosis/100.0*_TRAINING_PERC)
    n_validation = int(n_keratosis/100.0*_VALIDATION_PERC)
    training    += meta_keratosis[:n_training]
    validation  += meta_keratosis[n_training:n_training+n_validation]
    test        += meta_keratosis[n_training+n_validation:]
    # ...other
    n_other      = len(meta_other)
    n_training   = int(n_other/100.0*_TRAINING_PERC)
    n_validation = int(n_other/100.0*_VALIDATION_PERC)
    training    += meta_other[:n_training]
    validation  += meta_other[n_training:n_training+n_validation]
    test        += meta_other[n_training+n_validation:]

    # Reinsert extra cases
    def merge_test(extra_dict) :
      newcases = []
      for extra in extra_dict.values() :
        newcases.extend(extra)
      return newcases

    # Reinsert extra cases
    def merge_cases(sample_set, extra_list) :
      newcases = []
      for s in sample_set :
        if len(s[1])>0 :
          for extra in extra_list :
            match = extra.get(s[1], [])
            if len(match)>0 :
              newcases.extend(match)
              # Once assigned, the case shouldn't be reused
              extra[s[1]] = []
      sample_set.extend(newcases)

    merge_cases(test,       [extra_melanoma, extra_keratosis, extra_other])
    merge_cases(validation, [extra_melanoma, extra_keratosis, extra_other])
    merge_cases(training,   [extra_melanoma, extra_keratosis, extra_other])

    # Reshuffle each set
    random.shuffle(training)
    random.shuffle(validation)
    random.shuffle(test)

    # Contamination check
    training_ids   = set(( m[0] for m in training ))
    validation_ids = set(( m[0] for m in validation ))
    test_ids       = set(( m[0] for m in test ))

    if len(training_ids.intersection(validation_ids)) > 0 or \
       len(training_ids.intersection(test_ids)) > 0  or \
       len(validation_ids.intersection(test_ids)) > 0 :
      print('FATAL: cross contamination among sets. There are duplicate ids in source sets.', file=sys.stderr)
      print('tre & val: ', training_ids.intersection(validation_ids), file=sys.stderr)
      print('tre & tes: ', training_ids.intersection(test_ids), file=sys.stderr)
      print('val & tes: ', validation_ids.intersection(test_ids), file=sys.stderr)
      sys.exit(1)

    # Case contamination check --- we will be less strict about those but still emmit a warning
    training_ids   = set(( m[1] for m in training ))
    validation_ids = set(( m[1] for m in validation ))
    test_ids       = set(( m[1] for m in test ))
    trivial = set([''])
    if len(training_ids.intersection(validation_ids) - trivial) > 0 or \
       len(training_ids.intersection(test_ids) - trivial) > 0  or \
       len(validation_ids.intersection(test_ids)- trivial) > 0 :
      print('WARNING: case contamination among sets.', file=sys.stderr)
      print('tre & val: ', training_ids.intersection(validation_ids), file=sys.stderr)
      print('tre & tes: ', training_ids.intersection(test_ids), file=sys.stderr)
      print('val & tes: ', validation_ids.intersection(test_ids), file=sys.stderr)

    # Convert the training and validation sets.
    _convert_dataset('train', training, images_dir, dataset_dir)
    _convert_dataset('validation', validation, images_dir, dataset_dir)
    _convert_dataset('test', test, images_dir, dataset_dir)


  elif mode == 'TEST' :
    training   = []
    validation = []
    test       = metadata
    _convert_dataset('test', test, images_dir, dataset_dir)

    # This data is still needed by the prediction script
    CLASSES_TO_SIZES = { 'melanoma'  : 0,
                         'keratosis' : 0,
                         'other'     : len(test) }

  else :
    print('FATAL: invalid mode ', mode, file=sys.stderr)
    sys.exit(1)


  # Saves classes and split sizes
  print('INFO: ', CLASSES_TO_SIZES, file=sys.stderr)
  pickle.dump(CLASSES_TO_SIZES, open(os.path.join(dataset_dir, 'classes_to_sizes.pkl'), 'w'))

  SPLITS_TO_SIZES = { 'train'      : len(training),
                      'validation' : len(validation),
                      'test'       : len(test) }
  print('INFO: ', SPLITS_TO_SIZES,  file=sys.stderr)
  pickle.dump(SPLITS_TO_SIZES,  open(os.path.join(dataset_dir, 'splits_to_sizes.pkl'),  'w'))

  # Logs splits
  open(os.path.join(dataset_dir, 'train_set.log'), 'w').write('\n'.join([m[0] for m in training])+'\n')
  open(os.path.join(dataset_dir, 'valid_set.log'), 'w').write('\n'.join([m[0] for m in validation])+'\n')
  open(os.path.join(dataset_dir, 'test_set.log'),  'w').write('\n'.join([m[0] for m in test])+'\n')


if __name__=='__main__' :
  if (len(sys.argv)>4  and sys.argv[1]=='TEST') :
    run(mode=sys.argv[1], metadata_file=sys.argv[2], images_dir=sys.argv[3], dataset_dir=sys.argv[4])
  elif (len(sys.argv)>5  and sys.argv[1]=='TRAIN') :
    run(mode=sys.argv[1], metadata_file=sys.argv[2], images_dir=sys.argv[3], dataset_dir=sys.argv[4], blacklist_file=sys.argv[5])
  else :
    print('For preparing split train / validation / test for trainin phase:\n\n'
          'usage: convert_skin_lesion_dataset.py TRAIN metadata_file images_dir dataset_dir blacklist_file\n\n',
          'For preparing single file for test (without splitting or shuffling)\n'
          'usage: convert_skin_lesion_dataset.py TEST metadata_file images_dir dataset_dir', file=sys.stderr)
    sys.exit(1)

