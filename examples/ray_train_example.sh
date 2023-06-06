#!/bin/bash

# Example script to run a ray training experiment.
# You may wish to tune the resources provided to each Ray worker
# using the --gpu_per_worker and --cpu_per_worker flags, or increase
# workers with the --num_workers flag.

echo 'activating virtual environment'
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate tableshift

ulimit -u 127590 && python scripts/ray_train.py \
	--experiment adult \
	--num_samples 2 \
	--num_workers 1 \
	--cpu_per_worker 4 \
	--use_cached \
	--models node