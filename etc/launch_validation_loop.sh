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

if [ "$3" == "" ]; then
    echo usage: launch_validation_loop.sh [ INCEPTION | RESNET ] checkpoints_dir_path validation_dataset
    exit 1
fi

# Exit immediately on unitialized variables
set -u

if [ "$1" == "INCEPTION" ]; then
    MODEL=INCEPTION
elif [ "$1" == "RESNET" ]; then
    MODEL=RESNET
else
    echo Invalid model: "$1"
    exit 1
fi
CHECKPOINTS_DIR_PATH="$2"
VALIDATION_DATASET="$3"

ISBI_MODELS=$(dirname "$0")/..

BATCH=1     # It's not really worth to use larger batches : much more memory, not much faster time
REPLICAS=10 # Uses a smaller number of replica to validate (hopefully it doesn't change between models)

CHECKPOINT_REGEXP='\.ckpt-[0-9]*\.index'

function finish {
    # Delete temporary files
    rm "$TEMP_DIR/*.tmp" 2> /dev/null
    rmdir "$TEMP_DIR"
}
trap finish EXIT
TEMP_DIR=$(mktemp -d -t isbi.XXXXXXXXXX)

OLD_SCORES="$CHECKPOINTS_DIR_PATH/validation.log"
PREVIOUS_LIST="$TEMP_DIR/previouslist.tmp"
CURRENT_LIST="$TEMP_DIR/currentlist.tmp"
NEW_SCORES="$TEMP_DIR/newscores.tmp"
PREDICTIONS="$TEMP_DIR/predictions.tmp"
rm "$PREVIOUS_LIST" "$CURRENT_LIST" "$NEW_SCORES" "$PREDICTIONS" 2> /dev/null

echo Launching prediction for "$MODEL" on "$VALIDATION_DATASET"

echo -n > "$PREVIOUS_LIST"
while :
do
    ls -1 "$CHECKPOINTS_DIR_PATH" | grep "$CHECKPOINT_REGEXP" | sort > "$CURRENT_LIST"
    CHANGED=$(diff "$PREVIOUS_LIST" "$CURRENT_LIST")
    if [ "$CHANGED" != "" ]; then
        echo What changed: "$CHANGED" ....
        NEW_CHECKPOINT="$(ls -1 --sort=time "$CHECKPOINTS_DIR_PATH" | grep "$CHECKPOINT_REGEXP" | head -n 1)"
        echo Found new checkpoint: "$NEW_CHECKPOINT" --- launching validation...
        rm "$NEW_SCORES" 2> /dev/null
        if [ "$MODEL" == "INCEPTION" ]; then
            python "$ISBI_MODELS"/predict_image_classifier.py \
                --alsologtostderr \
                --checkpoint_path="$CHECKPOINTS_DIR_PATH" \
                --dataset_dir="$VALIDATION_DATASET" \
                --dataset_name=skin_lesions \
                --task_name=label \
                --dataset_split_name=validation \
                --model_name=inception_v4 \
                --preprocessing_name=dermatologic \
                --normalize_per_image=1 \
                --batch_size="$BATCH" \
                --eval_replicas="$REPLICAS" \
                --output_file="$PREDICTIONS" \
                --metrics_file="$NEW_SCORES"
        else
            python "$ISBI_MODELS"/predict_image_classifier.py \
                --alsologtostderr \
                --checkpoint_path="$CHECKPOINTS_DIR_PATH" \
                --dataset_dir="$VALIDATION_DATASET" \
                --dataset_name=skin_lesions \
                --task_name=label \
                --dataset_split_name=validation \
                --eval_image_size=224 \
                --model_name=resnet_v1_101 \
                --preprocessing_name=vgg \
                --batch_size="$BATCH" \
                --eval_replicas="$REPLICAS" \
                --output_file="$PREDICTIONS" \
                --metrics_file="$NEW_SCORES"
        fi
        NEW_SCORE=$(cut -d , -f 7 "$NEW_SCORES" | tr -d ' ' | tail -n 1)
        OLD_BEST=$(cut -d , -f 7 "$OLD_SCORES" | tr -d ' ' | sort -n | tail -n 1)
        [ "$OLD_BEST" == "" ]  && OLD_BEST="0"
        [ "$NEW_SCORE" == "" ] && NEW_SCORE=$OLD_BEST
        COMPARISON=$(echo $NEW_SCORE'>('$OLD_BEST'*1.001)' | bc -l)
        echo "$NEW_SCORE $OLD_BEST $COMPARISON"
        if [ "$COMPARISON" -eq 1 ]; then
            echo Improved validation --- saving model...
            NEW_FILE=$(tail -n 1 "$NEW_SCORES" | cut -d , -f 1)
            # Obliterates previous best
            mkdir -p "$CHECKPOINTS_DIR_PATH"/best
            rm "$CHECKPOINTS_DIR_PATH"/best/*
            ln "$NEW_FILE"* "$CHECKPOINTS_DIR_PATH"/best
            cp "$NEW_SCORES" "$CHECKPOINTS_DIR_PATH"/best/best.meta
            cp "$PREDICTIONS" "$CHECKPOINTS_DIR_PATH"/best/best.predictions
        fi
        tail -n 1 "$NEW_SCORES" >> "$OLD_SCORES"
        cp "$OLD_SCORES" "$CHECKPOINTS_DIR_PATH"/best/all.meta
        cp "$CURRENT_LIST" "$PREVIOUS_LIST"
    fi
    sleep 60
done


