#!/usr/bin/env bash

set -euo pipefail

DIR=$(dirname "$0")
source "$DIR"/config.sh

export CUDA_VISIBLE_DEVICES="0"

python3 word_bag/word_bag_tf.py \
        --cfg "$CONFIG" "$TEST_CONFIG" \
        --predict_data /path/to/your/data/shuf.wb.testing.* \
        --model_dir "$MODELDIR" \
	| jq -r '[.wght, .dbg, .lbl, .pred["1"]] | @csv' \
    | gzip -c > output.csv.gz
