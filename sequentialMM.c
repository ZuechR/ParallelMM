#include <assert.h>
#include <time.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>

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
    auto start_time_TR = std::chrono::steady_clock::now();
    matrix_transpose(dim_N, dim_O, mat_B, mat_BT);
    auto end_time_TR = std::chrono::steady_clock::now();
    free(mat_B);

    // Allocate space for C
    int* mat_C = (int*)malloc(dim_M * dim_O * sizeof(int));

    auto start_time_MM = std::chrono::steady_clock::now();
    sequential_transposed_MM(dim_M, dim_N, dim_O, mat_A, mat_BT, mat_C);
    auto end_time_MM = std::chrono::steady_clock::now();

    // printf("--- MATRIX C ---\n");
    // print_matrix_vector(dim_M, dim_O, mat_C);

    // Print elapsed time
    // Convert to milliseconds
    double elapsed_ms_TR =
        std::chrono::duration<double, std::milli>(end_time_TR - start_time_TR).count();

    double elapsed_ms_MM =
        std::chrono::duration<double, std::milli>(end_time_MM - start_time_MM).count();

    printf("Sequential transposition time is                %10.3f ms\n", elapsed_ms_TR);
    printf("Sequential MM computation time is               %10.3f ms\n", elapsed_ms_MM);

    // CLEAN-UP
    free(mat_A);
    free(mat_BT);
    free(mat_C);

    return 0;
}
