#!/bin/bash
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

# Exit immediately on errors or unitialized variables
set -e

if [ "$3" == "" ]; then
    echo usage: prepare_final_isbi_submission.sh test_dataset test_dataset_224 split_name submission_folder
    exit 1
fi

TEST_DATASET="$1"
TEST_DATASET_224="$2"
TEST_DATASET_SPLIT="$3"
SUBMISSION_FOLDER="$4"
ISBI_FOLDER=$(dirname "$0")/..
[ "$CHECKPOINT_ROOT_PATH" == "" ] && CHECKPOINT_ROOT_PATH="$ISBI_FOLDER"/running
[ "$SVM_MODEL_PATH" == "" ] && SVM_MODEL_PATH="$ISBI_FOLDER"/running/svm.models

set -u

[ -f "$SUBMISSION_FOLDER"/partial_rc25_1.txt ] || "$ISBI_FOLDER"/etc/predict_component_model_isbi.sh INCEPTION "$CHECKPOINT_ROOT_PATH"/checkpoints.rc25/model.ckpt-40000 "$TEST_DATASET" "$TEST_DATASET_SPLIT" "$SUBMISSION_FOLDER"/partial_rc25_1.txt --eval_replicas=50 --pool_scores=avg --normalize_per_image=1
[ -f "$SUBMISSION_FOLDER"/partial_rc25_2.txt ] || "$ISBI_FOLDER"/etc/predict_component_model_isbi.sh INCEPTION "$CHECKPOINT_ROOT_PATH"/checkpoints.rc25/model.ckpt-40000 "$TEST_DATASET" "$TEST_DATASET_SPLIT" "$SUBMISSION_FOLDER"/partial_rc25_2.txt --eval_replicas=50 --pool_scores=avg --normalize_per_image=1
[ -f "$SUBMISSION_FOLDER"/partial_rc25_3.txt ] || "$ISBI_FOLDER"/etc/predict_component_model_isbi.sh INCEPTION "$CHECKPOINT_ROOT_PATH"/checkpoints.rc25/model.ckpt-40000 "$TEST_DATASET" "$TEST_DATASET_SPLIT" "$SUBMISSION_FOLDER"/partial_rc25_3.txt --eval_replicas=50 --pool_scores=avg --normalize_per_image=1

[ -f "$SUBMISSION_FOLDER"/partial_rc25_max_1.txt ] || "$ISBI_FOLDER"/etc/predict_component_model_isbi.sh INCEPTION "$CHECKPOINT_ROOT_PATH"/checkpoints.rc25/model.ckpt-40000 "$TEST_DATASET" "$TEST_DATASET_SPLIT" "$SUBMISSION_FOLDER"/partial_rc25_max_1.txt --eval_replicas=50 --pool_scores=max --normalize_per_image=1
[ -f "$SUBMISSION_FOLDER"/partial_rc25_max_2.txt ] || "$ISBI_FOLDER"/etc/predict_component_model_isbi.sh INCEPTION "$CHECKPOINT_ROOT_PATH"/checkpoints.rc25/model.ckpt-40000 "$TEST_DATASET" "$TEST_DATASET_SPLIT" "$SUBMISSION_FOLDER"/partial_rc25_max_2.txt --eval_replicas=50 --pool_scores=max --normalize_per_image=1
[ -f "$SUBMISSION_FOLDER"/partial_rc25_max_3.txt ] || "$ISBI_FOLDER"/etc/predict_component_model_isbi.sh INCEPTION "$CHECKPOINT_ROOT_PATH"/checkpoints.rc25/model.ckpt-40000 "$TEST_DATASET" "$TEST_DATASET_SPLIT" "$SUBMISSION_FOLDER"/partial_rc25_max_3.txt --eval_replicas=50 --pool_scores=max --normalize_per_image=1

[ -f "$SUBMISSION_FOLDER"/partial_rc28_1.txt ] || "$ISBI_FOLDER"/etc/predict_component_model_isbi.sh INCEPTION "$CHECKPOINT_ROOT_PATH"/checkpoints.rc28/model.ckpt-40000 "$TEST_DATASET" "$TEST_DATASET_SPLIT" "$SUBMISSION_FOLDER"/partial_rc28_1.txt --eval_replicas=50 --pool_scores=avg --normalize_per_image=1
[ -f "$SUBMISSION_FOLDER"/partial_rc28_2.txt ] || "$ISBI_FOLDER"/etc/predict_component_model_isbi.sh INCEPTION "$CHECKPOINT_ROOT_PATH"/checkpoints.rc28/model.ckpt-40000 "$TEST_DATASET" "$TEST_DATASET_SPLIT" "$SUBMISSION_FOLDER"/partial_rc28_2.txt --eval_replicas=50 --pool_scores=avg --normalize_per_image=1
[ -f "$SUBMISSION_FOLDER"/partial_rc28_3.txt ] || "$ISBI_FOLDER"/etc/predict_component_model_isbi.sh INCEPTION "$CHECKPOINT_ROOT_PATH"/checkpoints.rc28/model.ckpt-40000 "$TEST_DATASET" "$TEST_DATASET_SPLIT" "$SUBMISSION_FOLDER"/partial_rc28_3.txt --eval_replicas=50 --pool_scores=avg --normalize_per_image=1

BEST_MODEL_RC30=$(ls -1 "$CHECKPOINT_ROOT_PATH"/checkpoints.rc30/best/model.ckpt-*.index)
BEST_MODEL_RC30="${BEST_MODEL_RC30%.index}"

[ -f "$SUBMISSION_FOLDER"/partial_rc30_1.txt ] || "$ISBI_FOLDER"/etc/predict_component_model_isbi.sh RESNET "$BEST_MODEL_RC30" "$TEST_DATASET_224" "$TEST_DATASET_SPLIT" "$SUBMISSION_FOLDER"/partial_rc30_1.txt --eval_replicas=50 --pool_scores=avg --normalize_per_image=1
[ -f "$SUBMISSION_FOLDER"/partial_rc30_2.txt ] || "$ISBI_FOLDER"/etc/predict_component_model_isbi.sh RESNET "$BEST_MODEL_RC30" "$TEST_DATASET_224" "$TEST_DATASET_SPLIT" "$SUBMISSION_FOLDER"/partial_rc30_2.txt --eval_replicas=50 --pool_scores=avg --normalize_per_image=1
[ -f "$SUBMISSION_FOLDER"/partial_rc30_3.txt ] || "$ISBI_FOLDER"/etc/predict_component_model_isbi.sh RESNET "$BEST_MODEL_RC30" "$TEST_DATASET_224" "$TEST_DATASET_SPLIT" "$SUBMISSION_FOLDER"/partial_rc30_3.txt --eval_replicas=50 --pool_scores=avg --normalize_per_image=1

[ -f "$SUBMISSION_FOLDER"/partial_rc25_50_s_1.txt ] || "$ISBI_FOLDER"/etc/features_component_model_isbi.sh INCEPTION "$CHECKPOINT_ROOT_PATH"/checkpoints.rc25/model.ckpt-40000 "$TEST_DATASET" "$TEST_DATASET_SPLIT" "$SUBMISSION_FOLDER"/feats.rc25_1 --eval_replicas=50 --pool_features=avg --pool_scores=avg --normalize_per_image=1 --add_scores_to_features=logits
[ -f "$SUBMISSION_FOLDER"/partial_rc25_50_s_1.txt ] || python "$ISBI_FOLDER"/predict_svm_layer.py --input_model "$SVM_MODEL_PATH"/rc25.50.svm  --input_test "$SUBMISSION_FOLDER"/feats.rc25_1 --output_predictions "$SUBMISSION_FOLDER"/partial_rc25_50_s_1.txt
[ -f "$SUBMISSION_FOLDER"/partial_rc25_50_s_2.txt ] || "$ISBI_FOLDER"/etc/features_component_model_isbi.sh INCEPTION "$CHECKPOINT_ROOT_PATH"/checkpoints.rc25/model.ckpt-40000 "$TEST_DATASET" "$TEST_DATASET_SPLIT" "$SUBMISSION_FOLDER"/feats.rc25_2 --eval_replicas=50 --pool_features=avg --pool_scores=avg --normalize_per_image=1 --add_scores_to_features=logits
[ -f "$SUBMISSION_FOLDER"/partial_rc25_50_s_2.txt ] || python "$ISBI_FOLDER"/predict_svm_layer.py --input_model "$SVM_MODEL_PATH"/rc25.50.svm  --input_test "$SUBMISSION_FOLDER"/feats.rc25_2 --output_predictions "$SUBMISSION_FOLDER"/partial_rc25_50_s_2.txt
[ -f "$SUBMISSION_FOLDER"/partial_rc25_50_s_3.txt ] || "$ISBI_FOLDER"/etc/features_component_model_isbi.sh INCEPTION "$CHECKPOINT_ROOT_PATH"/checkpoints.rc25/model.ckpt-40000 "$TEST_DATASET" "$TEST_DATASET_SPLIT" "$SUBMISSION_FOLDER"/feats.rc25_3 --eval_replicas=50 --pool_features=avg --pool_scores=avg --normalize_per_image=1 --add_scores_to_features=logits
[ -f "$SUBMISSION_FOLDER"/partial_rc25_50_s_3.txt ] || python "$ISBI_FOLDER"/predict_svm_layer.py --input_model "$SVM_MODEL_PATH"/rc25.50.svm  --input_test "$SUBMISSION_FOLDER"/feats.rc25_3 --output_predictions "$SUBMISSION_FOLDER"/partial_rc25_50_s_3.txt

[ -f "$SUBMISSION_FOLDER"/partial_rc28_50_s_1.txt ] || "$ISBI_FOLDER"/etc/features_component_model_isbi.sh INCEPTION "$CHECKPOINT_ROOT_PATH"/checkpoints.rc28/model.ckpt-40000 "$TEST_DATASET" "$TEST_DATASET_SPLIT" "$SUBMISSION_FOLDER"/feats.rc28_1 --eval_replicas=50 --pool_features=avg --pool_scores=avg --normalize_per_image=1 --add_scores_to_features=logits
[ -f "$SUBMISSION_FOLDER"/partial_rc28_50_s_1.txt ] || python "$ISBI_FOLDER"/predict_svm_layer.py --input_model "$SVM_MODEL_PATH"/rc28.50.svm  --input_test "$SUBMISSION_FOLDER"/feats.rc28_1 --output_predictions "$SUBMISSION_FOLDER"/partial_rc28_50_s_1.txt
[ -f "$SUBMISSION_FOLDER"/partial_rc28_50_s_2.txt ] || "$ISBI_FOLDER"/etc/features_component_model_isbi.sh INCEPTION "$CHECKPOINT_ROOT_PATH"/checkpoints.rc28/model.ckpt-40000 "$TEST_DATASET" "$TEST_DATASET_SPLIT" "$SUBMISSION_FOLDER"/feats.rc28_2 --eval_replicas=50 --pool_features=avg --pool_scores=avg --normalize_per_image=1 --add_scores_to_features=logits
[ -f "$SUBMISSION_FOLDER"/partial_rc28_50_s_2.txt ] || python "$ISBI_FOLDER"/predict_svm_layer.py --input_model "$SVM_MODEL_PATH"/rc28.50.svm  --input_test "$SUBMISSION_FOLDER"/feats.rc28_2 --output_predictions "$SUBMISSION_FOLDER"/partial_rc28_50_s_2.txt
[ -f "$SUBMISSION_FOLDER"/partial_rc28_50_s_3.txt ] || "$ISBI_FOLDER"/etc/features_component_model_isbi.sh INCEPTION "$CHECKPOINT_ROOT_PATH"/checkpoints.rc28/model.ckpt-40000 "$TEST_DATASET" "$TEST_DATASET_SPLIT" "$SUBMISSION_FOLDER"/feats.rc28_3 --eval_replicas=50 --pool_features=avg --pool_scores=avg --normalize_per_image=1 --add_scores_to_features=logits
[ -f "$SUBMISSION_FOLDER"/partial_rc28_50_s_3.txt ] || python "$ISBI_FOLDER"/predict_svm_layer.py --input_model "$SVM_MODEL_PATH"/rc28.50.svm  --input_test "$SUBMISSION_FOLDER"/feats.rc28_3 --output_predictions "$SUBMISSION_FOLDER"/partial_rc28_50_s_3.txt

[ -f "$SUBMISSION_FOLDER"/partial_rc28_50avg_s_1.txt ] || python "$ISBI_FOLDER"/predict_svm_layer.py --input_model "$SVM_MODEL_PATH"/rc28.50avg.svm  --input_test "$SUBMISSION_FOLDER"/feats.rc28_1 --output_predictions "$SUBMISSION_FOLDER"/partial_rc28_50avg_s_1.txt
[ -f "$SUBMISSION_FOLDER"/partial_rc28_50avg_s_2.txt ] || python "$ISBI_FOLDER"/predict_svm_layer.py --input_model "$SVM_MODEL_PATH"/rc28.50avg.svm  --input_test "$SUBMISSION_FOLDER"/feats.rc28_2 --output_predictions "$SUBMISSION_FOLDER"/partial_rc28_50avg_s_2.txt
[ -f "$SUBMISSION_FOLDER"/partial_rc28_50avg_s_3.txt ] || python "$ISBI_FOLDER"/predict_svm_layer.py --input_model "$SVM_MODEL_PATH"/rc28.50avg.svm  --input_test "$SUBMISSION_FOLDER"/feats.rc28_3 --output_predictions "$SUBMISSION_FOLDER"/partial_rc28_50avg_s_3.txt
