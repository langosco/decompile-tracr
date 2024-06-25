#!/bin/bash

trap "kill 0" SIGINT
#set -o errexit
export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS=cpu

CONFIG_NAME="small"
N_PARALLEL=10
#TERMSEQ="TERM,30000,KILL"
TERMSEQ="TERM,1000,KILL"

generate="python -m decompile_tracr.dataset.generate --config $CONFIG_NAME --ndata 99999999999"
compile="python -m decompile_tracr.dataset.compile --config $CONFIG_NAME"
compress="python -m decompile_tracr.dataset.compress --config small_compressed"

SEQ=$(seq 1 $N_PARALLEL)


parallel -n0 --timeout 3000 --ungroup --termseq $TERMSEQ "$generate" ::: $SEQ
python -m decompile_tracr.dataset.dedupe --config $CONFIG_NAME  && echo "Deduped programs."
parallel -n0 --timeout 10000 --ungroup --termseq $TERMSEQ "$compile" ::: $SEQ
python -m decompile_tracr.dataset.make_dataset --only_merge --config $CONFIG_NAME && echo "Merged dataset."
parallel -n0 --timeout 10000 --ungroup --termseq $TERMSEQ "$compress" ::: $SEQ
