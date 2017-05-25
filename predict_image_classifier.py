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
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os.path
import pickle
import tensorflow as tf

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory

import numpy as np
import sklearn.metrics
import sys

slim = tf.contrib.slim

_PREDICTION_OUTPUT_FORMAT='%.16f'

tf.app.flags.DEFINE_integer(
  'random_seed', 0, 'The random generator seed.')

tf.app.flags.DEFINE_integer(
  'batch_size', 1, 'The number of samples in each batch.')

tf.app.flags.DEFINE_string(
  'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
  'checkpoint_path', '/tmp/tfmodel/',
  'The directory where the model was written to or an absolute path to a '
  'checkpoint file.')

tf.app.flags.DEFINE_string(
  'eval_dir', '/tmp/tfmodel/', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
  'num_preprocessing_threads', 4,
  'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
  'dataset_name', 'skin_lesion', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
  'task_name', 'label', 'The name of the task to work (label, melanoma, or keratosis).')

tf.app.flags.DEFINE_string(
  'dataset_split_name', 'test', 'The name of the test split.')

tf.app.flags.DEFINE_string(
  'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
  'labels_offset', 0,
  'An offset for the labels in the dataset. This flag is primarily used to '
  'evaluate the VGG and ResNet architectures which do not use a background '
  'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
  'model_name', 'inception_v4', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
  'preprocessing_name', 'dermatologic', 'The name of the preprocessing to use. Default: dermatologic')

tf.app.flags.DEFINE_integer(
  'eval_image_size', None, 'Eval image size')

tf.app.flags.DEFINE_integer(
  'eval_replicas', 50, 'Number of replicas of the image to be evaluated. If >1 test augmentation.')

tf.app.flags.DEFINE_string(
  'id_field_name', None, 'The name of the field in the dataset metadata to identify the predictions.')

tf.app.flags.DEFINE_string(
  'output_file', None, 'File to output predictions or features, by default the standard output.')

tf.app.flags.DEFINE_string(
  'metrics_file', None, 'File to append metrics, in addition to the standard output.')

tf.app.flags.DEFINE_string(
  'output_format', 'text', 'Format of the output: text or (only with --extract_features) pickle.')

tf.app.flags.DEFINE_string(
  'pool_features', 'avg',
  'Function to pool the features across replicas: avg (default), max, xtrm, or none. '
  'If none, outputs one line per replica.'
  )

tf.app.flags.DEFINE_string(
  'pool_scores', 'avg',
  'Function to pool the probabilities across replicas: avg (default), max, avg_logits, max_logits, '
  'xtrm_logits, or none. If none, outputs one line per replica. On *_logits, the scores are pooled '
  'on the logits, before the softmax function is applied. If --extract_features is chosen only has '
  'effect if --add_scores_to_features is also chosen.'
  )

tf.app.flags.DEFINE_bool(
  'extract_features', False,
  'Extracts features instead of predictions to output_file. No metrics will be computed.')

tf.app.flags.DEFINE_string(
  'add_scores_to_features', 'none',
  'Adds model decisions at the end of the feature vector as part of it. Valid options are: '
  'none (default), probs, and logits. Must be used with --extract_features')

tf.app.flags.DEFINE_bool(
  'verbose_placement', False,
  'Shows detailed information about device placement.')

tf.app.flags.DEFINE_bool(
  'hard_placement', False,
  'Uses hard constraints for device placement on tensorflow sessions.')

tf.app.flags.DEFINE_bool(
  'fixed_memory', False,
  'Allocates the entire memory at once.')

tf.app.flags.DEFINE_bool(
    'aggressive_augmentation', False, 'Turn off fast_mode on preprocessing')

tf.app.flags.DEFINE_bool(
    'add_rotations', False, 'Add random rotations to augmentation on preprocessing')

tf.app.flags.DEFINE_integer(
    'normalize_per_image', 0, 'Normalization per image: 0 (None), 1 (Mean), 2 (Mean and Stddev)')

tf.app.flags.DEFINE_float(
    'minimum_area_to_crop', 0.05, 'Minimum area to keep in cropping for augmentation')

FLAGS = tf.app.flags.FLAGS

def main(_):
  if not FLAGS.dataset_dir :
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  if FLAGS.extract_features and FLAGS.metrics_file :
    raise ValueError('Option --metrics_file cannot be used with --extract_features')

  if FLAGS.pool_scores=='none' and FLAGS.metrics_file :
    raise ValueError('Option --metrics_file cannot be used without pooling')

  valid_decisions = [ 'none', 'probs', 'logits' ]
  if FLAGS.add_scores_to_features!='none' and not FLAGS.extract_features :
    raise ValueError('Option --add_scores_to_features must be used with --extract_features')
  elif not FLAGS.add_scores_to_features in valid_decisions :
    raise ValueError('Option --add_scores_to_features must be one of ' + ' '.join(valid_decisions))

  valid_score_poolings = [ 'avg', 'max', 'avg_logits', 'max_logits', 'xtrm_logits', 'none' ]
  if not FLAGS.pool_scores in valid_score_poolings :
    raise ValueError('Option --pool_scores must be one of ' + ' '.join(valid_poolings))

  if FLAGS.pool_features!='avg' and not FLAGS.extract_features:
    raise ValueError('Option --pool_features must be used with --extract_features')

  valid_feature_poolings = [ 'avg', 'max', 'xtrm', 'none' ]
  if not FLAGS.pool_features in valid_feature_poolings :
    raise ValueError('Option --pool_features must be one of ' + ' '.join(valid_feature_poolings))

  if (FLAGS.extract_features and FLAGS.add_scores_to_features!='none' and
      (FLAGS.pool_scores!=FLAGS.pool_features) and
      (FLAGS.pool_scores=='none' or FLAGS.pool_features=='none')) :
    raise ValueError('Option --pool_features=none requires --pool_scores=none and vice-versa when extracting both features and decisions')

  valid_output_formats = [ 'text', 'pickle' ]
  if not FLAGS.output_format in valid_output_formats :
    raise ValueError('Option --output_format must be one of ' + ' '.join(valid_output_formats))

  if FLAGS.output_format=='pickle' and not FLAGS.extract_features :
    raise ValueError('Option --output_format=pickle requires --extract_features')

  if FLAGS.output_format=='pickle' and not FLAGS.output_file :
    raise ValueError('Option --output_format=pickle requires --output_file')

  if not FLAGS.normalize_per_image in [0, 1, 2] :
    raise ValueError('Invalid value for --normalize_per_image: must be 0, 1 or 2')

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    tf_global_step = slim.get_or_create_global_step()

    ######################
    # Select the dataset #
    ######################
    dataset = dataset_factory.get_dataset(
        FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

    ####################
    # Select the model #
    ####################
    num_classes = (dataset.num_classes - FLAGS.labels_offset)
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=num_classes,
        is_training=False)

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle=False,
        common_queue_capacity=64,
        common_queue_min=0)

    if FLAGS.id_field_name :
      field_id = FLAGS.id_field_name
    else :
      field_id = FLAGS.task_name
    [image, image_id, label] = provider.get(['image', field_id, FLAGS.task_name])
    label -= FLAGS.labels_offset

    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=FLAGS.eval_replicas>1)

    eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size

    def preprocess(img) :
      if FLAGS.preprocessing_name=='dermatologic' :
        return image_preprocessing_fn(img, eval_image_size, eval_image_size,
                bbox=None,
                fast_mode=not FLAGS.aggressive_augmentation,
                area_range=(FLAGS.minimum_area_to_crop, 1.0),
                add_rotations=FLAGS.add_rotations,
                normalize_per_image=FLAGS.normalize_per_image)
      else :
        return image_preprocessing_fn(img, eval_image_size, eval_image_size)

    if FLAGS.eval_replicas>1 :
      aug_list = []
      for r in range(FLAGS.eval_replicas) :
        aug_list.append(preprocess(image))
    else :
      aug_list = [ preprocess(image) for r in range(FLAGS.eval_replicas) ]

    image_aug  = tf.pack(aug_list)

    ####################
    # Define the model #
    ####################
    logits, end_points = network_fn(image_aug)
    if FLAGS.model_name[:6]=='resnet' :
      logits = tf.squeeze(logits, [1, 2])

    variables_to_restore = slim.get_variables_to_restore()

    def tf_reduce_maxabs_axis_0(tensor) :
      xtrm_rows  = tf.argmax(tf.abs(tensor), axis=0)
      xtrm_cols  = tf.range(num_classes, dtype=tf.int64)
      # For other axes the indexing below must change
      xtrm_index = tf.transpose(tf.stack([xtrm_rows, xtrm_cols]))
      return tf.gather_nd(tensor, xtrm_index)

    pooled_features = True
    nada = tf.constant([float('nan')])
    if FLAGS.extract_features :
      features = end_points['PreLogitsFlatten']
      # Pools across replicas
      if FLAGS.pool_features == 'avg' :
        features = tf.reduce_mean(features, axis=0)
      elif FLAGS.pool_features == 'max' :
        features = tf.reduce_max(features, axis=0)
      elif FLAGS.pool_features == 'xtrm' :
        features = tf_reduce_maxabs_axis_0(features)
      elif FLAGS.pool_features == 'none' :
        pooled_features = False
      else :
        assert False, "Invalid FLAGS.pool_features: '%s'" % FLAGS.pool_features
      feature_size = int(features.get_shape()[0 if pooled_features else 1])
    else :
      features = nada
      feature_size = 0

    pooled_scores = True
    # Pools across replicas
    if FLAGS.pool_scores == 'avg' :
      probabilities = tf.nn.softmax(logits)
      probabilities = tf.reduce_mean(probabilities, axis=0)
      logits_out    = tf.reduce_mean(logits, axis=0)
    elif FLAGS.pool_scores == 'avg_logits' :
      logits_out    = tf.reduce_mean(logits, axis=0)
      probabilities = tf.nn.softmax(logits_out)
    elif FLAGS.pool_scores == 'max' :
      probabilities = tf.nn.softmax(logits)
      probabilities = tf.reduce_max(probabilities, axis=0)
      logits_out    = tf.reduce_max(logits, axis=0)
    elif FLAGS.pool_scores == 'max_logits' :
      logits_out    = tf.reduce_max(logits, axis=0)
      probabilities = tf.nn.softmax(logits_out)
    elif FLAGS.pool_scores == 'xtrm_logits' :
      logits_out    = tf_reduce_maxabs_axis_0(logits)
      probabilities = tf.nn.softmax(logits_out)
    elif FLAGS.pool_scores == 'none' :
      probabilities = tf.nn.softmax(logits)
      logits_out    = logits
      pooled_scores = False
    else :
      assert False, "Invalid FLAGS.pool_scores: '%s'" % FLAGS.pool_scores

    assert pooled_scores==pooled_features

    # Predicts across classes (on each replica, if not pooled)
    predictions = tf.argmax(probabilities, axis=0 if pooled_scores else 1)

    ###########################
    # Performs the prediction #
    ###########################

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
      checkpoint_path = FLAGS.checkpoint_path

    tf.logging.info('Evaluating %s' % checkpoint_path)

    session_config = tf.ConfigProto(
        log_device_placement = FLAGS.verbose_placement,
        allow_soft_placement = not FLAGS.hard_placement)
    if not FLAGS.fixed_memory :
      session_config.gpu_options.allow_growth=True

    # This ensures that we make a single pass over all of the data.
    num_samples = dataset.num_samples

    init_fn = slim.assign_from_checkpoint_fn(checkpoint_path,
      variables_to_restore) # slim.get_model_variables(FLAGS.model_name))

    outfile = open(FLAGS.output_file, 'w') if FLAGS.output_file else sys.stdout
    num_outputs = num_samples if pooled_features else num_samples * FLAGS.eval_replicas
    tensor_id = image_id if FLAGS.id_field_name else nada
    if FLAGS.extract_features :
      # Features - control message and targets
      openP  = '{'
      closeP = '}'
      if FLAGS.add_scores_to_features == 'probs' :
        feature_size += num_classes
        targets =[ tensor_id, label, features, probabilities, nada ]
      elif FLAGS.add_scores_to_features == 'logits' :
        feature_size += num_classes
        targets =[ tensor_id, label, features, logits_out, nada ]
      else :
        targets =[ tensor_id, label, features, nada, nada ]
      # Features - outputs header
      if FLAGS.output_format=='text' :
        print(num_outputs, file=outfile)
        header  = [ FLAGS.id_field_name ] if FLAGS.id_field_name else  [ ]
        header += [ 'truth' ]
        header += [ 'feature[%d]' % feature_size ]
        print(', '.join(header), file=outfile)
      else :
        pickle.dump([num_outputs, feature_size, FLAGS.__flags], outfile)
      # Features - outputs contents
      def print_replica(image_id, label, feats) :
        if FLAGS.output_format=='text' :
          record  = [ image_id ] if FLAGS.id_field_name else  [ ]
          record += [ str(label) ]
          record += [ _PREDICTION_OUTPUT_FORMAT % feats[f]  for f in range(feature_size) ]
          print(', '.join(record), file=outfile)
        else :
          pickle.dump([image_id, label, feats], outfile)
    else : # => FLAGS.extract_features==False
      if pooled_scores :
        list_ids         = []
        list_labels      = []
        list_scores      = []
        list_predictions = []
      # Predictions - control message and targets
      openP  = '['
      closeP = ']'
      targets =[ tensor_id, label, nada, probabilities, predictions ]
      # Predictions - print header
      header  = [ FLAGS.id_field_name ] if FLAGS.id_field_name else  [ ]
      header += [ 'truth' ]
      header += [ 'class%d' % c for c in range(num_classes) ] if dataset.labels_to_names is None else \
                [ dataset.labels_to_names[c] + '[%d]' % c for c in range(num_classes) ]
      header += [ 'prediction' ]
      print(', '.join(header), file=outfile)
      # Predictions - print contents
      def print_replica(image_id, label, scores, pred) :
        record  = [ image_id ] if FLAGS.id_field_name else  [ ]
        record += [ str(label) ]
        record += [ _PREDICTION_OUTPUT_FORMAT % scores[c] for c in range(num_classes) ]
        record += [ str(pred) ]
        print(', '.join(record), file=outfile)

    with tf.Session(config=session_config) as sess:
      init_fn(sess)

      # init_op = tf.global_variables_initializer()
      # sess.run(init_op)

      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)
      count_changes = 0
      count_disagreements = 0

      for s in range(num_samples) :
        print(openP, end='', file=sys.stderr)
        next_id, next_lab, next_feats, next_scores, next_preds = sess.run(
            targets, options=tf.RunOptions(timeout_in_ms=120000))
        next_id = next_id if FLAGS.id_field_name else ''
        if FLAGS.extract_features :
          if FLAGS.add_scores_to_features!='none' :
            next_feats = np.concatenate((next_feats, next_scores), axis=0 if pooled_features else 1)
          if pooled_features :
            print_replica(next_id, next_lab, next_feats)
          else :
            for r in range(FLAGS.eval_replicas) :
              print_replica(next_id, next_lab, next_feats[r])
        else :
          if pooled_scores :
            list_ids.append(next_id)
            list_labels.append(next_lab)
            list_scores.append(next_scores)
            list_predictions.append(next_preds)
            print_replica(next_id, next_lab, next_scores, next_preds)
          else :
            for r in range(FLAGS.eval_replicas) :
              print_replica(next_id, next_lab, next_scores[r], next_preds[r])
        print(closeP, end='\n' if (s+1) % 40 == 0 else '', file=sys.stderr)
        # print('{All variables: ', len (tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)), '}')
      print('', file=sys.stderr)

      coord.request_stop()
      coord.join(threads)

  if pooled_scores and not FLAGS.extract_features :

    if FLAGS.metrics_file :
      metfile = open(FLAGS.metrics_file, 'w')
      print('checkpoint, acc, auc[1], auc[2], map[1], map[2], isbi', file=metfile)
    else :
      metfile = None

    np_labels        = np.asarray(list_labels)
    np_predictions   = np.asarray(list_predictions)
    np_probabilities = np.asarray(list_scores)

    metfile and print(checkpoint_path, ', ' , sep='', end='', file=metfile)
    print('Confusion Matrix:\n', sklearn.metrics.confusion_matrix(np_labels, np_predictions))
    m_acc = sklearn.metrics.accuracy_score(np_labels, np_predictions)
    print('Acc: ', m_acc)
    metfile and print(m_acc, ', ' , sep='', end='', file=metfile)

    try :
      aucs = []
      mAPs = []
      for j in range(num_classes) :
        np_labels_j = np.int64(np_labels == j)
        np_scores_j = np_probabilities[:, j]
        # fpr, tpr, _ = sklearn.metrics.roc_curve(np_labels_j, np_scores_j)
        auc = sklearn.metrics.roc_auc_score(np_labels_j, np_scores_j)
        aucs.append(auc)
        print('AUC[%d]: ' % j, auc)
        # pre, rec, _ = sklearn.metrics.precision_recall_curve(np_labels_j, np_scores_j)
        mAP = sklearn.metrics.average_precision_score(np_labels_j, np_scores_j)
        mAPs.append(mAP)
        print('mAP[%d]: ' % j, mAP)
      metfile and print(aucs[1], ', ' , sep='', end='', file=metfile)
      metfile and print(aucs[2], ', ' , sep='', end='', file=metfile)
      metfile and print(mAPs[1], ', ' , sep='', end='', file=metfile)
      metfile and print(mAPs[2], ', ' , sep='', end='', file=metfile)
      metfile and print((aucs[1]+aucs[2])/2.0, sep='', end='\n', file=metfile)
      print('AUCavg: ', sum(aucs) / num_classes)
      print('mAPavg: ', sum(mAPs) / num_classes)
    except ValueError :
      pass

if __name__ == '__main__':
  tf.app.run()
