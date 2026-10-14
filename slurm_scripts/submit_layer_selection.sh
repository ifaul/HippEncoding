#!/bin/bash
#SBATCH --job-name=whisper_layer_selection
#SBATCH --error=/lustre/project/bdeen/HippEncoding/logs/whisper_layer_%A_%a.err
#SBATCH --partition=centos7
#SBATCH --time=15:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=22-32

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

# Calculate layer from array ID
LAYER=$SLURM_ARRAY_TASK_ID

python3.11 select_layer.py \
    --subjects UTS03 \
    --sessions 1 2 3 4 5 \
    --modelname whisper-large \
    --layer $LAYER \
    --trim 5 \
    --ndelays 4 \
    --max-stories 6 \
    --use-corr