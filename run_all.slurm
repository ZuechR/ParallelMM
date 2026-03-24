#!/bin/bash
#SBATCH --job-name=mm
#SBATCH --output=logs/mm_%A_%a.out
#SBATCH --error=logs/mm_%A_%a.err
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --array=0-3

sizes=(8192 16384 32768 65536)
procs=(1 2 4 8)
threads=(2 4 8 16)

n_runs=3

PARALLEL_DIR="experiments/parallel"
SEQUENTIAL_DIR="experiments/sequential"

mkdir -p "$PARALLEL_DIR" "$SEQUENTIAL_DIR" logs

# Select size for this array task
n=${sizes[$SLURM_ARRAY_TASK_ID]}

echo "Array task $SLURM_ARRAY_TASK_ID handling size n=$n"

##################################
# PARALLEL EXPERIMENTS FOR THIS SIZE
##################################

SIZE_DIR="$PARALLEL_DIR/$n"
mkdir -p "$SIZE_DIR"

for p in "${procs[@]}"; do
    for t in "${threads[@]}"; do

        export OMP_NUM_THREADS=$t
        EXPERIMENT_DIR="$SIZE_DIR/${n}_${p}_${t}"
        mkdir -p "$EXPERIMENT_DIR"

        for run in $(seq 1 $n_runs); do
            out_file="$EXPERIMENT_DIR/run_${run}.txt"

            echo "Running parallel n=$n p=$p t=$t run=$run"
            srun --ntasks=$p --cpus-per-task=$t ./build/parallelMM "$n" "$n" "$n" > "$out_file"
        done
    done
done

##################################
# SEQUENTIAL EXPERIMENTS FOR THIS SIZE
##################################

SIZE_DIR="$SEQUENTIAL_DIR/$n"
mkdir -p "$SIZE_DIR"

for run in $(seq 1 $n_runs); do
    out_file="$SIZE_DIR/run_${run}.txt"

    echo "Running sequential n=$n run=$run"
    srun --ntasks=1 --cpus-per-task=1 ./build/sequentialMM "$n" "$n" "$n" > "$out_file"
done