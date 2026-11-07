#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --requeue
# Runs line $SLURM_ARRAY_TASK_ID (0-based) of the jobs file $1; submit.sh and sweep.sh set the rest.
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$1")
echo "[$(hostname) $(date)] $line"
eval "$line"
