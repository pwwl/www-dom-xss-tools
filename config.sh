#!/usr/bin/env bash

OUTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

MODELDIR="$OUTDIR"/model_500

CONFIGDIR="$OUTDIR"/configs

CONFIG="$CONFIGDIR"/data8_model500.yaml
TRAIN_CONFIG="$CONFIGDIR"/data8_train_config.yaml
TEST_CONFIG="$CONFIGDIR"/data8_test_config.yaml
PERF_CONFIG="$CONFIGDIR"/data8_perf_test.yaml
