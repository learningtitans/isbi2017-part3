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

# Exit immediately on errors
set -e

if [ "$5" == "" ]; then
    echo usage: features_component_model_isbi.sh [ INCEPTION | RESNET ] checkpoint_path test_dataset split_name features_file
    exit 1
fi

set -u # Unitialized variables are errors, don't default to empty

if [ "$1" == "INCEPTION" ]; then
    MODEL=INCEPTION
elif [ "$1" == "RESNET" ]; then
    MODEL=RESNET
else
    echo Invalid model: "$1"
    exit 1
fi
CHECKPOINT_PATH="$2"
TEST_DATASET="$3"
TEST_DATASET_SPLIT="$4"
FEATURES_FILE="$5"

ISBI_MODELS=$(dirname "$0")/..

BATCH=1 # It's not really worth to use larger batches : much more memory, not much faster time

# function finish {
#     # Delete temporary files
#     rm "$TEMP_DIR"/*.tmp 2> /dev/null
#     rmdir "$TEMP_DIR"
# }
# trap finish EXIT
# TEMP_DIR=`mktemp -d -t isbi.XXXXXXXXXX`

shift 5

set +u
DATASET_NAME=$(basename "$TEST_DATASET")
if [ "$1" == "" ]; then
    echo Launching prediction for "$DATASET_NAME" on model "$MODEL"
else
    echo Launching prediction for "$DATASET_NAME" on model "$MODEL" with parameters "$@"
fi
set -u

if [ "$MODEL" == "INCEPTION" ]; then
    python "$ISBI_MODELS"/predict_image_classifier.py \
        --alsologtostderr \
        --checkpoint_path="$CHECKPOINT_PATH"  \
        --dataset_dir="$TEST_DATASET" \
        --dataset_name=skin_lesions \
        --task_name=label \
        --dataset_split_name="$TEST_DATASET_SPLIT" \
        --model_name=inception_v4 \
        --preprocessing_name=dermatologic \
        --id_field_name=id \
        --batch_size="$BATCH" \
        --extract_features \
        --output_file="$FEATURES_FILE" \
        --output_format=pickle \
        "$@"
else
    python "$ISBI_MODELS"/predict_image_classifier.py \
        --alsologtostderr \
        --checkpoint_path="$CHECKPOINT_PATH" \
        --dataset_dir="$TEST_DATASET" \
        --dataset_name=skin_lesions \
        --task_name=label \
        --dataset_split_name="$TEST_DATASET_SPLIT" \
        --model_name=resnet_v1_101 \
        --preprocessing_name=vgg \
        --eval_image_size=224 \
        --id_field_name=id \
        --batch_size="$BATCH" \
        --extract_features \
        --output_file="$FEATURES_FILE" \
        --output_format=pickle \
        "$@"
fi
