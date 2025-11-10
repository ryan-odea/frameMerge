#!/bin/bash
#SBATCH --job-name=merge_frames
#SBATCH --output=logs/merge_all_%j.out
#SBATCH --error=logs/merge_all_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=day
#SBATCH --time=06:00:00

# ==========================
#  USER CONFIGURATION
# ==========================
LIST_FILE="../u_crystfel/01_find_beam_center/runs.lst"   # Input list of full .h5 paths
N_MERGED_FRAMES=3
DATA_LOCATION="entry/data"
DATA_NAME="data"
OUTDIR="merged_outputs"
N_JOBS=8      # How many merges to run concurrently

export LIST_FILE N_MERGED_FRAMES DATA_LOCATION DATA_NAME OUTDIR

source ~/.bashrc
module load anaconda
module load parallel
conda activate general

mkdir -p "$OUTDIR" logs

run() {
    INPUT_FILE="$1"

    BASENAME=$(basename "$INPUT_FILE" .h5)

    for BIN_TAG in 011 101 110; do
        case "$BIN_TAG" in
            011) SKIP_INDEX=0 ;;
            101) SKIP_INDEX=1 ;;
            110) SKIP_INDEX=2 ;;
        esac

        OUTPUT_FILE="${OUTDIR}/${BASENAME}_${BIN_TAG}.h5"

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting ${BASENAME}_${BIN_TAG}.h5"

        python -m frameMerge.main \
            -f "$INPUT_FILE" \
            -o "$OUTPUT_FILE" \
            --n_frames 45 \
            --n_merged_frames "$N_MERGED_FRAMES" \
            --skip_pattern "$SKIP_INDEX" \
            --data_location "$DATA_LOCATION" \
            --data_name "$DATA_NAME" \
            --n_workers 8

        EXIT_CODE=$?
        if [[ $EXIT_CODE -ne 0 ]]; then
            echo "Error: $INPUT_FILE (pattern $BIN_TAG) failed with exit code $EXIT_CODE" >&2
        fi
    done
}
export -f run

cat "$LIST_FILE" | parallel -j "$N_JOBS" run {}

