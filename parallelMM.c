#include <assert.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "utils.h"

const int MASTER = 0;

/**
 * This gets executed by every process
 *
 * @param rank is the rank (id) of the process
 * @param n_processes is the number of processes running
 * ...
 */
void parallel_MM(int rank, int n_processes, unsigned int dim_M, unsigned int dim_N, unsigned int dim_O, int mat_A[], int mat_BT[], int mat_C[]) {
    // Storing time points for time tracking
    double start_total_time, start_comm_time, start_comp_time, start_gather_time, end_time;

    start_total_time = MPI_Wtime();

    // Helpers definitions
    // rows[rank] = # of rows of A (and C) this process handles
    // counts{A,C}[rank] = # of values of A (or C) this process handles
    // displs{A,C}[rank] = offset of this processes values in A (or C)
    // Recall that the idea is to assign at each process p a set of adjacent rows A_p of A and gather from it (into the
    // master) a set of adjacent rows C_p of C.
    int rows[n_processes], counts_A[n_processes], displs_A[n_processes], counts_C[n_processes], displs_C[n_processes];

    // If (M % n_processes) == 0, then it is divisible and rows[i] is always M/n_processes
    // Otherwise (M/n_processes + 1) to the the first (M % n_processes) processes and (int, truncated so it's a floor operation) M/n_processes to
    // the others
    const unsigned int frac = dim_M / n_processes;
    for (unsigned int i = 0; i < n_processes; i++) {
        // If M % n_processes == 0, then i < 0 is never true, so it will always assign frac to rows[i]
        rows[i] = i < (dim_M % n_processes) ? frac + 1 : frac;

        // counts and displs for A and C
        counts_A[i] = rows[i] * dim_N;
        counts_C[i] = rows[i] * dim_O;
        displs_A[i] = i > 0 ? counts_A[i - 1] + displs_A[i - 1] : 0;
        displs_C[i] = i > 0 ? counts_C[i - 1] + displs_C[i - 1] : 0;
    }

    // Scatter A and C from the master to all other processes as sets of adjacent rows A_p and C_p
    int A_p_size = counts_A[rank];
    int C_p_size = A_p_size / dim_N * dim_O;
    assert(A_p_size > 0);   // This in teory never happens given the guard in main (n_processes > dim_M)
    assert(C_p_size > 0);
    int* A_p = (int*)malloc(A_p_size * sizeof(int));
    int* C_p = (int*)malloc(C_p_size * sizeof(int));

    start_comm_time = MPI_Wtime();

    // Broadcast BT from the master to all other processes
    MPI_Bcast(mat_BT, dim_N * dim_O, MPI_INT, 0, MPI_COMM_WORLD);

    // Scattering of A rows into sets of adjacents rows to each process
    MPI_Scatterv(mat_A, counts_A, displs_A, MPI_INT, A_p, counts_A[rank], MPI_INT, 0, MPI_COMM_WORLD);

    start_comp_time = MPI_Wtime();

    // Compute C_p as the product of A_p and BT
    sequential_transposed_MM(rows[rank], dim_N, dim_O, A_p, mat_BT, C_p);

    start_gather_time = MPI_Wtime();

    // Gather all C_p from each process p into C in the master
    MPI_Gatherv(C_p, C_p_size, MPI_INT, mat_C, counts_C, displs_C, MPI_INT, 0, MPI_COMM_WORLD);

    end_time = MPI_Wtime();

    // If we're the master process we need to print the results
    if (rank == MASTER) {
        // Communication time
        printf("Parallel MM communication time is               %10.3f ms\n",
               ((start_comp_time - start_comm_time) + (end_time - start_gather_time)) * 1.e3);

        // Computation time
        printf("Parallel MM computation time is                 %10.3f ms\n", (start_gather_time - start_comp_time) * 1.e3);

        // Communication + computation time
        printf("Parallel MM communication + computation time is %10.3f ms\n", (end_time - start_comm_time) * 1.e3);

        // Total time for the process to execute on the master
        printf("Parallel MM total time is                       %10.3f ms\n", (end_time - start_total_time) * 1.e3);
    }

    // CLEAN-UP
    free(A_p);
    free(C_p);
}

/**
 * The general idea is to define master and slave processes, where the master has the role to initialize matrix A and B,
 * in order to then distribute the computation over the other processes.
 */
int main(int argc, char** argv) {
    if (argc < 4) {
        printf("Wrong arguments! Should be M, N, O [seed].");
        return -1;
    }

    // Get process_rank and n_processes
    int process_rank, n_processes;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &n_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

    // Parse dimensions
    int dim_M = atoi(argv[1]);
    int dim_N = atoi(argv[2]);
    int dim_O = atoi(argv[3]);
    // The same seed guarantees the same matrices across runs and across parallel/sequential executables
    unsigned int seed = argc > 4 ? (unsigned int)atoi(argv[4]) : (unsigned int)time(NULL);

    assert(dim_M > 0);
    assert(dim_N > 0);
    assert(dim_O > 0);

    // Prevent that the scatter creates processes with no rows (malloc and MPI_Scatterv/Gatherv behaviours with size 0 are undefined)
    if (n_processes > dim_M) {
        if (process_rank == MASTER) fprintf(stderr, "Error: n_processes (%d) > dim_M (%d)\n", n_processes, dim_M);
        MPI_Finalize();
        return MPI_ERR_ARG;
    }

    // Define matrixes
    int* mat_A = (int*)malloc(dim_M * dim_N * sizeof(int));
    int* mat_BT = (int*)malloc(dim_N * dim_O * sizeof(int));  // B transposed

    // If this is the master we need to populate the matrices
    if (process_rank == MASTER) {
        int* mat_B = (int*)malloc(dim_N * dim_O * sizeof(int));

        // Populate Matrices
        srand(seed);
        populate_matrix_as_vector(dim_M, dim_N, mat_A);
        populate_matrix_as_vector(dim_N, dim_O, mat_B);

        // printf("--- MATRIX A ---\n");
        // print_matrix_vector(dim_M, dim_N, mat_A);
        // printf("--- MATRIX B ---\n");
        // print_matrix_vector(dim_N, dim_O, mat_B);

        // Transpose B
        // TODO: Unless we are measuring also the transpose time (which we currently aren't) we could assume B is already the transposed version
        matrix_transpose(dim_N, dim_O, mat_B, mat_BT);
        free(mat_B);
    }

    // Allocate space for C
    int* mat_C = (int*)malloc(dim_M * dim_O * sizeof(int));

    // Matrix multiplication parallel computation (this also prints elapsed times)
    parallel_MM(process_rank, n_processes, dim_M, dim_N, dim_O, mat_A, mat_BT, mat_C);

    if (process_rank == MASTER) {
        // printf("--- MATRIX C ---\n");
        // print_matrix_vector(dim_M, dim_O, mat_C);

        // Integrity check on the result
        int* check_C = (int*)malloc(dim_M * dim_O * sizeof(int));
        sequential_transposed_MM(dim_M, dim_N, dim_O, mat_A, mat_BT, check_C);
        for (unsigned int i = 0; i < dim_M * dim_O; i++) {
            assert(mat_C[i] == check_C[i]);
        }
        free(check_C);
    }

    // CLEAN-UP
    free(mat_A);
    free(mat_BT);
    free(mat_C);

    return MPI_Finalize();
}
