#!/bin/bash
#SBATCH --job-name=mm
#SBATCH --output=logs/mm_%j.out
#SBATCH --error=logs/mm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=5
#SBATCH --time=120:00:00

# ADJUST NTASKS AND CPUS-PER-TASK IN THE LINES ABOVE TO MATCH THE MAXIMUM NUMBER OF PROCESSES AND THREADS

# select the grid of sizes, number of processes and number of threads
sizes=(1024 2048 4096 8192 16384)
procs=(1 2 3 4 5)
threads=(1 2 3 4 5)

n_runs = 5 
# number of runs to average over for each configuration

PARALLEL_DIR="experiments/parallel"
SEQUENTIAL_DIR="experiments/sequential"

mkdir -p "$PARALLEL_DIR" logs

# RUN PARALLEL EXPERIMENTS
for n in "${sizes[@]}"; do
    SIZE_DIR="$PARALLEL_DIR/$n"
    mkdir -p "$SIZE_DIR"
    for p in "${procs[@]}"; do
        for t in "${threads[@]}"; do

            # skip oversubscription if needed
            # if (( p * t > 25 )); then
            #    continue
            # fi

            export OMP_NUM_THREADS=$t
            EXPERIMENT_DIR ="$SIZE_DIR/${n}_${p}_${t}"

            # parallel/1024/1024_1_1/run_1.txt
            for run in $(seq 1 $n_runs); do
                out_file="$EXPERIMENT_DIR/run_${run}.txt"

                echo "Running parallel n=$n p=$p t=$t run $run"
                srun --ntasks=$p --cpus-per-task=$t ./build/parallelMM "$n" "$n" "$n" > "$out_file"

            done
            
        done
    done
done

##################################

mkdir -p "$SEQUENTIAL_DIR"

# RUN SEQUENTIAL EXPERIMENTS
for n in "${sizes[@]}"; do

    SIZE_DIR="$SEQUENTIAL_DIR/$n"    
    mkdir -p "$SIZE_DIR"

    # sequential/1024/run_1.txt
    for run in $(seq 1 $n_runs); do
        out_file="$SIZE_DIR/run_${run}.txt"

        echo "Running sequential n=$n run $run"
        srun --ntasks=1 --cpus-per-task=1 ./build/sequentialMM "$n" "$n" "$n" > "$out_file"

    done 
done