#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 100;
    int local_N = N / size;
    int *full_array = NULL;
    int *local_array = (int*)malloc(local_N * sizeof(int));

    if (rank == 0) {
        full_array = (int*)malloc(N * sizeof(int));
        for (int i = 0; i < N; i++) full_array[i] = i + 1;
    }

    MPI_Scatter(full_array, local_N, MPI_INT, local_array, local_N, MPI_INT, 0, MPI_COMM_WORLD);

    int local_sum = 0;
    for (int i = 0; i < local_N; i++) local_sum += local_array[i];

    int global_sum = 0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Global sum: %d (expected 5050)\n", global_sum);
    }
    free(local_array);
    if (rank == 0) free(full_array);
    MPI_Finalize();
    return 0;
}
