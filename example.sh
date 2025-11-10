#!/bin/bash
#SBATCH --job-name=merge_frames
#SBATCH --output=logs/merge_all_%j.out
#SBATCH --error=logs/merge_all_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --partition=day
#SBATCH --ntasks=1
#SBATCH --nodes=1

LIST_FILE="../u_crystfel/01_find_beam_center/runs.lst"           # Full paths to input .h5 files
N_MERGED_FRAMES=3                   # Merge 3 frames each time
DATA_LOCATION="entry/data"          # HDF5 group path
DATA_NAME="data"                    # Dataset name inside HDF5
OUTDIR="merged_outputs"             # Where to put merged results

source ~/.bashrc
module load anaconda
conda activate general

mkdir -p "$OUTDIR" logs
declare -A SKIP_PATTERNS
SKIP_PATTERNS["011"]="0"
SKIP_PATTERNS["101"]="1"
SKIP_PATTERNS["110"]="2"

while read -r INPUT_FILE; do
    [[ -z "$INPUT_FILE" ]] && continue  # skip blank lines

    BASENAME=$(basename "$INPUT_FILE" .h5)
    DIRNAME=$(dirname "$INPUT_FILE")

    for BIN_TAG in "${!SKIP_PATTERNS[@]}"; do
        SKIP_INDEX=${SKIP_PATTERNS[$BIN_TAG]}
        OUTPUT_FILE="${OUTDIR}/${BASENAME}_${BIN_TAG}.h5"

        srun --exclusive -N1 -n1 \
            python -m frameMerge.main \
            -f "$INPUT_FILE" \
            -o "$OUTPUT_FILE" \
            --n_frames 45 \
            --n_merged_frames "$N_MERGED_FRAMES" \
            --skip_pattern "$SKIP_INDEX" \
            --data_location "$DATA_LOCATION" \
            --data_name "$DATA_NAME" \
            --n_workers 8 &
    done
done < "$LIST_FILE"
wait
