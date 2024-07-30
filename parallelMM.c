#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

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

void parallel_mult(int rank, int size, int M, int N, int O, int A[], int BT[], int C[])
{
    double sTotal, eTotal, sComm, sComp, eComp;
    sTotal = MPI_Wtime();

    // Definition of the number of rows of A (and C) assigned to each process as the vector rows, as well as the counts
    // and displacements vectors definitions
    // Recall that the idea is to assign at each process p a set of adjacent rows A_p of A and gather from it (into the
    // master) a set of adjacent rows C_p of C.
    int rows[size], countsA[size], displsA[size], countsC[size], displsC[size];

    // if M%size == 0, then it is divisible and rows[i] is always M/size
    // Otherwise (M/size + 1) to the the first M % size processes and (int) M/size to the others
    int frac = M/size;
    for (int i = 0; i < size; i++)
    {
        if (M % size == 0)
            rows[i] = frac;
        else
            rows[i] = i < M % size ? frac + 1 : frac;
        // counts and displs for A
        countsA[i] = rows[i] * N;
        displsA[i] = i > 0 ? countsA[i] + displsA[i - 1] : 0;
        // counts and displs for C
        countsC[i] = rows[i] * O;
        displsC[i] = i > 0 ? countsC[i] + displsC[i - 1] : 0;
    }

    // Scatter A from the master to all other processes as sets of adjacent rows A_p

    int A_p_size = countsA[rank]; // the size of A_p follows from above
    int* A_p = malloc(A_p_size * sizeof(int)); // buffer used to store the portion of A recived from the master

    int C_p_size = A_p_size/N * O;
    int* C_p = malloc(C_p_size * sizeof(int)); // buffer used to store C_p

    sComm = MPI_Wtime();

    // Broadcast BT from the master to all other processes
    MPI_Bcast(BT, N*O, MPI_INT, 0, MPI_COMM_WORLD);

    // Scattering of A rows into sets of adjacents rows to each process
    MPI_Scatterv(A,	countsA, displsA, MPI_INT, A_p, countsA[rank], MPI_INT, 0, MPI_COMM_WORLD);

    sComp = MPI_Wtime();

    // Compute C_p as the product of A_p and B (BT since we use the transposed algorithm)
    sequentialTransposeMM(rows[rank], N, O, A_p, BT, C_p);

    eComp = MPI_Wtime();

    // Gather all C_p from each process p into C in the master
    MPI_Gatherv(C_p, C_p_size, MPI_INT, C, countsC, displsC, MPI_INT, 0, MPI_COMM_WORLD);

    eTotal = MPI_Wtime();

    if(rank == MASTER)
    {
        // Total time for the process to execute on the master
        printf("Parallel MM total time is %f ms\n", (eTotal - sTotal) * 1.e3);

        // Communication time
        printf("Parallel MM communication time is %f ms\n", ((sComp - sComm) + (eTotal - eComp)) * 1.e3);

        // Computation time
        printf("Parallel MM computation time is %f ms\n", (eComp - sComp) * 1.e3);

        // Communication + computation time
        printf("Parallel MM communication + computation time is %f ms\n", (eTotal - sComm) * 1.e3);
    }

    // CLEAN-UP
    free(A_p);
    free(C_p);
}

/* The general idea is to define master and slave processes, where the master has the role to initialize matrix A and B,
 * in order to then distribute the computation over the other processes.
 */

int main(int argc, char **argv) {
    if(argc < 4)
    {
        printf("Wrong arguments! Should be M, N, O.");
        return -1;
    }

    int rank, size;
	MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int O = atoi(argv[3]);

    // Define matrixes
    int* A = malloc(M*N*sizeof(int));
    int* BT = malloc(N*O*sizeof(int)); // B transposed

    if(rank == MASTER)
    {
        // printf("M %d, N %d, O %d, \n", M, N, O);
        int* B = malloc(N*O*sizeof(int));
        // printf("\nMatrix A of size %d x %d: \n", M, N);
        populateMatrixAsVector(M, N, A);
        // printf("\nMatrix B of size %d x %d: \n", N, O);
        populateMatrixAsVector(N, O, B);

        // printf("\nMatrix BT of size %d x %d: \n", O, N);
        matrix_transpose(N, O, B, BT);
        free(B);
    }

    int* C = malloc(M*O*sizeof(int));

    // PARALLEL EXECUTION
    parallel_mult(rank, size, M, N, O, A, BT, C);

    /*
    if(rank == MASTER)
    {
        printf("\nMatrix C of size %d x %d: \n", M, O);
        printMatrixVector(M, O, C);
    }

    // SEQUENTIAL EXECUTION

    if(rank == MASTER)
    {
        double start, end;
        int firstEl = C[0];
        start = MPI_Wtime();
        sequentialTransposeMM(M, N, O, A, BT, C);
        end = MPI_Wtime();
        // printf("\nMatrix C of size %d x %d: \n", M, O);
        // printMatrixVector(M, O, C);
        printf("\nSequential MM computation time is %f ms\n", (end - start) * 1.e6);
        if(firstEl != C[0])
            printf("Something went wrong! Sequential and parallel results do not match!");
    }
    */

    // CLEAN-UP
    free(A);
    free(BT);
    free(C);
    MPI_Finalize();
    return 0;
}
