#!/bin/bash
#SBATCH --job-name=CLIP_embed
#SBATCH --output=logs/CLIP_%A_%a.out
#SBATCH --error=logs/CLIP_%A_%a.err
#SBATCH --partition=centos7
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=82

cd /lustre/project/bdeen/HippEncoding

# Activate your virtual environment / conda environment
source /share/apps/anaconda/3/5.1.0/etc/profile.d/conda.sh
conda activate /lustre/project/bdeen/HippEncoding/conda_envs/hippocampus

# Redirect temporary files to Lustre space
export TMPDIR="/lustre/project/bdeen/HippEncoding/tmp"
export PIP_CACHE_DIR="/lustre/project/bdeen/HippEncoding/pip_cache"

# Thread management for PyTorch CPU parallelization
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_WAIT_POLICY=PASSIVE

# Shared Hugging Face cache directory to avoid downloading whisper-large multiple times
export HF_HOME="/lustre/project/bdeen/HippEncoding/HuggingFace/hub"

# Read the narrative name corresponding to this SLURM Task ID
STORY=$(sed -n "${SLURM_ARRAY_TASK_ID}p" narratives.txt)

echo "Starting task $SLURM_ARRAY_TASK_ID on $(hostname) for story: $STORY"

# Run the Python script for the single narrative
python3.11 deep-fMRI-dataset/encoding/CLIP_embedding.py \
    --modelname clip-large \
    --stories "$STORY" \
    --device cpu \
    --cache-dir "$HF_HOME"

echo "Finished processing narrative: $STORY"
