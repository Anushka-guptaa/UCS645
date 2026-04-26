#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N (1<<20)   // 65536 as per assignment

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double a = 3.14;
    double *X = (double*)malloc(N * sizeof(double));
    double *Y = (double*)malloc(N * sizeof(double));

    // Initialize on rank 0 and scatter
    if (rank == 0) {
        for (int i = 0; i < N; i++) {
            X[i] = i * 0.1;
            Y[i] = i * 0.2;
        }
    }

    int local_n = N / size;
    double *local_X = (double*)malloc(local_n * sizeof(double));
    double *local_Y = (double*)malloc(local_n * sizeof(double));

    MPI_Scatter(X, local_n, MPI_DOUBLE, local_X, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(Y, local_n, MPI_DOUBLE, local_Y, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);   // sync all processes
double start = MPI_Wtime();

// DAXPY computation
for (int i = 0; i < local_n; i++) {
    local_X[i] = a * local_X[i] + local_Y[i];
}

MPI_Barrier(MPI_COMM_WORLD);   // ensure all finished
double end = MPI_Wtime();

    // Gather back (just for verification)
    MPI_Gather(local_X, local_n, MPI_DOUBLE, X, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("DAXPY completed successfully!\n");
        printf("Time taken with %d processes: %f seconds\n", size, end - start);
        printf("Speedup can be measured by comparing np=1 vs np=4/8\n");
    }

    free(local_X); free(local_Y); free(X); free(Y);
    MPI_Finalize();
    return 0;
}
