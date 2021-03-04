#!/usr/bin/env bash

set -eou pipefail

DIR=$(dirname "$0")
source "$DIR"/config.sh

ODIR="training-data"
DATA="$ODIR/shuf.wb.training"

function train_mini_epoch {
    
    for i in $(seq "$2" "$3" | shuf)
    do
        echo "training on file $i...."
        python3 word_bag/word_bag_tf.py \
            --cfg "$CONFIG" "$TRAIN_CONFIG" \
            --training_data "$DATA".$i \
            --model_dir "$MODELDIR"
    done

    cp -r "$MODELDIR" "$MODELDIR"-checkpoint."$1"."$2"-"$3"

}

export CUDA_VISIBLE_DEVICES="0"
train_mini_epoch "1" "100" "124"
train_mini_epoch "1" "125" "149"
train_mini_epoch "1" "150" "174"
train_mini_epoch "1" "175" "199"
train_mini_epoch "1" "200" "224"
train_mini_epoch "1" "225" "249"
train_mini_epoch "1" "250" "274"
train_mini_epoch "1" "275" "299"


