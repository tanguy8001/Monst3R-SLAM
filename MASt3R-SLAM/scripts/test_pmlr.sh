#!/bin/bash
#SBATCH --account=pmlr
#SBATCH --time=02:00:00   # Request 2 hours
#SBATCH --partition=jobs
#SBATCH --output=out/test_pmlr_%j.out

echo "Starting job at $(date)"

# Run a long sleep to simulate long computation
sleep 7200  # 7200 seconds = 2 hours

echo "Job finished at $(date)"
