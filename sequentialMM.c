#include <assert.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include "utils.h"

int main(int argc, char** argv) {
    if (argc < 4) {
        printf("Wrong arguments! Should be M, N, O [seed].");
        return -1;
    }

    // Parse dimensions
    int dim_M = atoi(argv[1]);
    int dim_N = atoi(argv[2]);
    int dim_O = atoi(argv[3]);
    // The same seed guarantees the same matrices across runs and across parallel/sequential executables
    unsigned int seed = argc > 4 ? (unsigned int)atoi(argv[4]) : (unsigned int)time(NULL);

    assert(dim_M > 0);
    assert(dim_N > 0);
    assert(dim_O > 0);

    // Define matrixes
    int* mat_A = (int*)malloc(dim_M * dim_N * sizeof(int));
    int* mat_B = (int*)malloc(dim_N * dim_O * sizeof(int));
    int* mat_BT = (int*)malloc(dim_N * dim_O * sizeof(int));  // B transposed

    // Populate Matrices
    srand(seed);
    populate_matrix_as_vector(dim_M, dim_N, mat_A);
    populate_matrix_as_vector(dim_N, dim_O, mat_B);

    // printf("--- MATRIX A ---\n");
    // print_matrix_vector(dim_M, dim_N, mat_A);
    // printf("--- MATRIX B ---\n");
    // print_matrix_vector(dim_N, dim_O, mat_B);

    // Transpose B
    // Transpose B
    double start_time_TR = omp_get_wtime();
    matrix_transpose(dim_N, dim_O, mat_B, mat_BT);
    double end_time_TR = omp_get_wtime();
    free(mat_B);

    // Allocate space for C
    int* mat_C = (int*)malloc(dim_M * dim_O * sizeof(int));

    double start_time_MM = omp_get_wtime();
    sequential_transposed_MM(dim_M, dim_N, dim_O, mat_A, mat_BT, mat_C);
    double end_time_MM = omp_get_wtime();

    // printf("--- MATRIX C ---\n");
    // print_matrix_vector(dim_M, dim_O, mat_C);

    // Print elapsed time
    // Convert to milliseconds
    double elapsed_ms_TR = (end_time_TR - start_time_TR) * 1e3;

    double elapsed_ms_MM = (end_time_MM - start_time_MM) * 1e3;

    printf("Sequential transposition time is                %10.3f ms\n", elapsed_ms_TR);
    printf("Sequential MM computation time is               %10.3f ms\n", elapsed_ms_MM);

    // CLEAN-UP
    free(mat_A);
    free(mat_BT);
    free(mat_C);

    return 0;
}
