#include <stdio.h>
#include <stdlib.h>

/**
 * Prints to stdout the matrix
 */
void print_matrix_vector(const unsigned int rows, const unsigned int columns, int matrix[]) {
    for (unsigned int i = 0; i < rows; i++) {
        for (unsigned int j = 0; j < columns; j++) {
            printf("%7d ", matrix[(i * columns) + j]);
        }
        printf("\n");
    }
}

/**
 * Populate the matrix with random values
 */
void populate_matrix_as_vector(const unsigned int rows, const unsigned int columns, int matrix[]) {
    const int limit = 100;
    for (unsigned int i = 0; i < rows; i++) {
        for (unsigned int j = 0; j < columns; j++) {
            const int value = (rand() % (2 * limit)) - limit;  // Signed int in [-100,+100)
            matrix[(i * columns) + j] = value;
        }
    }
}

/**
 * Transpose a matrix
 */
void matrix_transpose(unsigned int rows, unsigned int columns, int mat[], int mat_T[]) {
    for (unsigned int i = 0; i < rows; i++) {
        for (unsigned int j = 0; j < columns; j++) {
            mat_T[i + (j * rows)] = mat[(i * columns) + j];
        }
    }
}

/**
 * Compute the matrix multiplication between A and an already transposed B
 */
void sequential_transposed_MM(const unsigned int dim_M, const unsigned int dim_N, const unsigned int dim_O, int mat_A[], int mat_BT[], int mat_C[]) {
    for (unsigned int i = 0; i < dim_M; i++) {
        for (unsigned int j = 0; j < dim_O; j++) {
            int sum = 0;
            for (unsigned int k = 0; k < dim_N; k++) {
                sum += mat_A[(i * dim_N) + k] * mat_BT[(j * dim_N) + k];
            }
            mat_C[(i * dim_O) + j] = sum;
        }
    }
}
