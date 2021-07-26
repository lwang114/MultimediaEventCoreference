#!/bin/bash

start_stage=0
stop_stage=100

## Weight fusion model
if [ $start_stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  python train.py --config configs/config_joint_coref_video_m2e2.json
fi

## Vector fusion model
if [ $start_stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  python train_transfer.py --config configs/config_coref_transfer_mmaction_finetuned_oneie_video_m2e2.json
fi

## Supervised text-only model
if [ $start_stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  python train_transfer.py --config configs/config_coref_transfer_textonly_video_m2e2.json
fi

## Amodal SMT
if [ $start_stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  cwd=$(pwd)
  cd unsupervised
  python amodal_smt_event_coreferencer.py --config ../configs/config_amodal_smt_aligner_video_m2e2.json
  cd ${cwd}
fi

## Text SMT
if [ $start_stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  cwd=$(pwd)
  cd unsupervised
  python text_only_smt_event_coreferencer.py --config ../configs/config_amodal_smt_aligner_video_m2e2.json
  cd ${cwd}
fi

## V->T SMT
if [ $start_stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  cwd=$(pwd)
  cd unsupervised
  python image_to_text_event_aligner.py --config ../configs/config_image_to_text_smt_video_m2e2.json
  cd ${cwd}
fi

## HDP
if [ $start_stage -le 6 ] && [ $stop_stage -ge 6 ]; then
  cwd=$(pwd)
  cd unsupervised
  python finite_hdp_event_aligner.py --config ../configs/config_hdp_video_m2e2.json
  cd ${cwd}
fi

## DDCRP
if [ $start_stage -le 7 ] && [ $stop_stage -ge 7 ]; then
  cwd=$(pwd)
  cd unsupervised
  python finite_ddcrp_event_aligner.py --config ../configs/config_ddcrp_video_m2e2.json
  cd ${cwd}
fi
