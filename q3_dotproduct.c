#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 50000000   // 50 million (reduced from 500M for Colab memory)

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double multiplier = 2.5;
    if (rank == 0) {
        printf("Enter scaling multiplier (default 2.5): ");
        // For simplicity we hardcode; you can scanf if you want
    }
    MPI_Bcast(&multiplier, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int local_n = N / size;
    double local_dot = 0.0;
    double start = MPI_Wtime();

    // Each process generates its own chunk
    for (int i = 0; i < local_n; i++) {
        double a = 1.0;
        double b = 2.0 * multiplier;
        local_dot += a * b;
    }

    double global_dot = 0.0;
    MPI_Reduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double end = MPI_Wtime();

    if (rank == 0) {
        printf("Final Dot Product: %f\n", global_dot);
        printf("Time with %d processes: %f seconds\n", size, end - start);
        printf("Run with np=1,2,4,8 to calculate speedup & efficiency\n");
    }

    MPI_Finalize();
    return 0;
}
