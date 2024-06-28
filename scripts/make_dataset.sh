#!/bin/bash

export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS=cpu

BASE_CONFIG="small"
COMP_CONFIG="small_compressed"
N_PARALLEL=5
TERMSEQ="TERM,30000,TERM,5000,KILL,25"

generate="python -m decompile_tracr.dataset.generate --config $BASE_CONFIG --ndata 99999999999"
compile="python -m decompile_tracr.dataset.compile --config $BASE_CONFIG"
compress="python -m decompile_tracr.dataset.compress --config $COMP_CONFIG"

SEQ=$(seq 1 $N_PARALLEL)


parallel -n0 --timeout 2000 --ungroup --termseq $TERMSEQ "$generate" ::: $SEQ
python -m decompile_tracr.dataset.dedupe --config $BASE_CONFIG
parallel -n0 --timeout 20000 --ungroup --termseq $TERMSEQ "$compile" ::: $SEQ
python -m decompile_tracr.dataset.make_dataset --merge --add_ids --config $BASE_CONFIG
parallel -n0 --timeout 20000 --ungroup --termseq $TERMSEQ "$compress" ::: $SEQ
python -m decompile_tracr.dataset.make_dataset --merge --config $COMP_CONFIG
