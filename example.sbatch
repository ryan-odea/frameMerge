#!/bin/bash
#SBATCH --job-name=merge_frames
#SBATCH --output=logs/merge_all_%j.out
#SBATCH --error=logs/merge_all_%j.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --partition=day
#SBATCH --time=12:00:00

# ==========================
#  USER CONFIGURATION
# ==========================
LIST_FILE="../u_crystfel/01_find_beam_center/runs.lst"
N_MERGED_FRAMES=3
DATA_LOCATION="entry/data"
DATA_NAME="data"
OUTDIR="merged_outputs"
N_FILES=8
N_SKIP=3
WORKERS_PER_MERGE=8

export LIST_FILE N_MERGED_FRAMES DATA_LOCATION DATA_NAME OUTDIR WORKERS_PER_MERGE

source ~/.bashrc
module load anaconda
module load parallel
conda activate general

mkdir -p "$OUTDIR" logs

declare -A SKIP_PATTERNS
SKIP_PATTERNS["011"]="0"
SKIP_PATTERNS["101"]="1"
SKIP_PATTERNS["110"]="2"

run_merge() {
    local INPUT_FILE="$1"
    local BIN_TAG="$2"
    local BASENAME=$(basename "$INPUT_FILE" .h5)
    local SKIP_INDEX=${SKIP_PATTERNS[$BIN_TAG]}

    OUTPUT_FILE="${OUTDIR}/${BASENAME}_${BIN_TAG}.h5"

    python -m frameMerge.main \
        -f "$INPUT_FILE" \
        -o "$OUTPUT_FILE" \
        --n_frames 45 \
        --n_merged_frames "$N_MERGED_FRAMES" \
        --skip_pattern "$SKIP_INDEX" \
        --data_location "$DATA_LOCATION" \
        --data_name "$DATA_NAME" \
        --n_workers "$WORKERS_PER_MERGE"
}
export -f run_merge
export SKIP_PATTERNS

run_file() {
    local INPUT_FILE="$1"
    parallel -j "$N_SKIP" run_merge "$INPUT_FILE" {} ::: 011 101 110
}
export -f run_file

cat "$LIST_FILE" | parallel -j "$N_FILES" run_file {}
