#include <stdio.h>
#include <stdlib.h>
#include <time.h>

const int MASTER = 0;

void printMatrixVector(const int rows, const int columns, int v[])
{
    for(int i = 0; i < rows; i++)
    {
        for(int j = 0; j < columns; j++)
        {
            printf("%d ", v[i*columns+j]);
        }
        printf("\n");
    }
}

void populateMatrixAsVector(const int rows, const int columns, int v[])
{
    for(int i = 0; i < rows; i++)
        for(int j = 0; j < columns; j++)
        {
            const short sign = rand() % 2 == 0 ? -1 : 1; // produce a random sign
            v[i*columns + j] = sign * (rand() % 100);
        }
    // printMatrixVector(rows, columns, v);
}

void matrix_transpose(int rows, int columns, int B[], int BT[]) {
    for(int i = 0; i < rows; i++)
    {
        for(int j = 0; j < columns; j++)
        {
            BT[i + j*rows] = B[i*columns + j];
        }
    }
    // printMatrixVector(columns, rows, BT);
}


void sequentialTransposeMM(const int M, const int N, const int O, int A[], int BT[], int C[])
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < O; j++) {
            int sum = 0;
            for (int k = 0; k < N; k++) {
                sum += A[i*N + k] * BT[j*N + k];
            }
            C[i * O + j] = sum;
        }
    }
}

int main(int argc, char **argv) {
    if(argc < 4)
    {
        printf("Wrong arguments! Should be M, N, O.");
        return -1;
    }

    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int O = atoi(argv[3]);

    // Define matrixes
    int* A = malloc(M*N*sizeof(int));
    int* BT = malloc(N*O*sizeof(int)); // B transposed

    // printf("M %d, N %d, O %d, \n", M, N, O);
    int* B = malloc(N*O*sizeof(int));
    // printf("\nMatrix A of size %d x %d: \n", M, N);
    populateMatrixAsVector(M, N, A);
    // printf("\nMatrix B of size %d x %d: \n", N, O);
    populateMatrixAsVector(N, O, B);
    // printf("\nMatrix BT of size %d x %d: \n", O, N);
    matrix_transpose(N, O, B, BT);
    free(B);

    int* C = malloc(M*O*sizeof(int));

    double start, end;
    start = clock();
    sequentialTransposeMM(M, N, O, A, BT, C);
    end = clock();
    // printf("\nMatrix C of size %d x %d: \n", M, O);
    // printMatrixVector(M, O, C);
    printf("Sequential MM computation time is %.3f ms\n", ((double) (end - start)) * 1000 / CLOCKS_PER_SEC);

    // CLEAN-UP
    free(A);
    free(BT);
    free(C);
    return 0;
}
