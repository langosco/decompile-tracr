#!/bin/bash

trap "kill 0" SIGINT
set -o errexit
CONFIG_NAME="default"
N_PARALLEL=20


for i in $(seq 1 $N_PARALLEL)
do
    python -m decompile_tracr.dataset.generate --disable_tqdm --config $CONFIG_NAME --ndata 100 &
done
wait
echo "Done generating programs"


python -m decompile_tracr.dataset.dedupe --config $CONFIG_NAME


for i in $(seq 1 $N_PARALLEL)
do
#    python -m decompile_tracr.dataset.compile_and_compress --config $CONFIG_NAME &
    python -m decompile_tracr.dataset.compile --config $CONFIG_NAME &
done
wait
echo "Done compiling programs"


python -m decompile_tracr.dataset.make_dataset --only_to_h5 --make_test_splits --config $CONFIG_NAME
echo "Saved dataset to h5."
