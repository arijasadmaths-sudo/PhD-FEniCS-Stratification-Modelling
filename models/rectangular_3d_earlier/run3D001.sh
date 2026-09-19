#!/bin/bash

#SBATCH --job-name=fenics3D_efficient_hdiv
#SBATCH --account=YOUR_ACCOUNT

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1

#SBATCH --time=175:00:00
#SBATCH --mem-per-cpu=4G

#SBATCH --output=fenics3D_efficient_hdiv_%j.out
#SBATCH --error=fenics3D_efficient_hdiv_%j.err

module purge
module add languages/python/fenics-2019.1.0

echo "========================================"
echo "3D rectangular restartable efficient H(div) simulation"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks: $SLURM_NTASKS"
echo "Start: $(date)"
echo "========================================"

srun --mpi=pmi2 python3 run3D001.py

echo "========================================"
echo "Finished: $(date)"
echo "========================================"
