#!/bin/bash

start_stage=0
stop_stage=100

## Effect of number of clusters for amodal SMT
if [ $start_stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  cwd=$(pwd)
  cd unsupervised
  for k in 10 15 30 60 120; do 
    python amodal_smt_event_coreferencer.py --config ../configs/config_amodal_smt_aligner_video_m2e2_${k}clusters.json
  done
  cd ${cwd}
fi

