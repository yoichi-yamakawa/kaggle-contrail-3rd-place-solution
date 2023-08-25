#!/bin/bash

cfg_list=(
    # exp164-maxvit-large-512-crop-5e4-gc
    exp166-maxvit-base-512-crop-5e4-25ep-gc
)

for EXP_NAME in "${cfg_list[@]}"
do 
    for i in 0
    do
        python scripts/training/run.py run --config_path=scripts/training/config/${EXP_NAME}.yaml \
        -resume=False num_workers=12 fold_index=$i debug=True gpus=1 train_batch_size=1
    done
done