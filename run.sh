#!/bin/bash

SAVE_ROOT="./saved_models"
RESULT_ROOT="./results"

 python main.py \
     --mode train \
     --classes bottle capsule cable carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper\
     --save_root $SAVE_ROOT \
     --result_root $RESULT_ROOT \
     --lr 0.0003 \
     --epochs 200 \
     --batch_size 4 \
     --iteration 3 \
     --eval_iteration 3 \
     --eval_epoch 1 \
     --device cuda:0