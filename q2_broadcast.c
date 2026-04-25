#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define ARRAY_SIZE 10000000   // 10 million doubles (~80 MB)

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double *data = (double*)malloc(ARRAY_SIZE * sizeof(double));
    double start, end;

    if (rank == 0) printf("=== Running with %d processes ===\n", size);

    // ------------------ Part A: MyBcast (linear loop) ------------------
    if (rank == 0) {
        for (int i = 0; i < ARRAY_SIZE; i++) data[i] = i * 0.1;
        start = MPI_Wtime();
        for (int i = 1; i < size; i++) {
            MPI_Send(data, ARRAY_SIZE, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(data, ARRAY_SIZE, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        end = MPI_Wtime();
        printf("MyBcast (linear) time: %f seconds\n", end - start);
    }

    // ------------------ Part B: MPI_Bcast ------------------
    if (rank == 0) {
        for (int i = 0; i < ARRAY_SIZE; i++) data[i] = i * 0.1;
        start = MPI_Wtime();
    }
    MPI_Bcast(data, ARRAY_SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        end = MPI_Wtime();
        printf("MPI_Bcast time: %f seconds\n", end - start);
    }

    free(data);
    MPI_Finalize();
    return 0;
}
