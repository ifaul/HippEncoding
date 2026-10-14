#!/bin/bash
#SBATCH --job-name=whisper_encoding
#SBATCH --error=logs/whisper_encoding_%A_%a.err
#SBATCH --partition=centos7
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=1-9

cd /lustre/project/bdeen/HippEncoding/deep-fMRI-dataset/encoding

# Load system modules
module load gcc

# Activate virtual environment / conda environment
source /share/apps/anaconda/3/5.1.0/etc/profile.d/conda.sh
conda activate /lustre/project/bdeen/HippEncoding/conda_envs/hippocampus

# Set Threading Variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Map SLURM_ARRAY_TASK_ID to subject ID
SUBJECTS=("UTS01" "UTS02" "UTS03" "UTS04" "UTS05" "UTS06" "UTS07" "UTS08" "UTS09")
SUBJECT=${SUBJECTS[$SLURM_ARRAY_TASK_ID - 1]}

python3.11 whisper_encoding.py \
    --subject $SUBJECT \
    --sessions 1 2 3 4 5 \
    --feature whisper \
    --modelname whisper-large \
    --layer 17 \
    --backend numpy \
    --n-jobs -1 \
    --omp-threads 16 \
    --use-corr