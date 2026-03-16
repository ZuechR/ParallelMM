#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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
    struct timespec start_time_TR;
    struct timespec end_time_TR;

    clock_gettime(CLOCK_MONOTONIC, &start_time_TR);
    matrix_transpose(dim_N, dim_O, mat_B, mat_BT);
    clock_gettime(CLOCK_MONOTONIC, &end_time_TR);
    free(mat_B);

    // Allocate space for C
    int* mat_C = (int*)malloc(dim_M * dim_O * sizeof(int));

    // Measuring time
    struct timespec start_time_MM;
    struct timespec end_time_MM;

    // Matrix multiplication sequential computation
    clock_gettime(CLOCK_MONOTONIC, &start_time_MM);
    sequential_transposed_MM(dim_M, dim_N, dim_O, mat_A, mat_BT, mat_C);
    clock_gettime(CLOCK_MONOTONIC, &end_time_MM);

    // printf("--- MATRIX C ---\n");
    // print_matrix_vector(dim_M, dim_O, mat_C);

    // Print elapsed time
    double elapsed_ms_TR = ((end_time_TR.tv_sec - start_time_TR.tv_sec) * 1e3) + ((end_time_TR.tv_nsec - start_time_TR.tv_nsec) / 1e6);
    double elapsed_ms_MM = ((end_time_MM.tv_sec - start_time_MM.tv_sec) * 1e3) + ((end_time_MM.tv_nsec - start_time_MM.tv_nsec) / 1e6);
    printf("Sequential transposition time is                %10.3f ms\n", elapsed_ms_TR);
    printf("Sequential MM computation time is               %10.3f ms\n", elapsed_ms_MM);

    // CLEAN-UP
    free(mat_A);
    free(mat_BT);
    free(mat_C);

    return 0;
}
