#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    srand(time(NULL) + rank);

    int local_max = -1, local_min = 1001;
    for (int i = 0; i < 10; i++) {
        int num = rand() % 1001;
        if (num > local_max) local_max = num;
        if (num < local_min) local_min = num;
    }

    int local_max_pair[2] = {local_max, rank};
    int local_min_pair[2] = {local_min, rank};
    int global_max_pair[2], global_min_pair[2];

    MPI_Reduce(local_max_pair, global_max_pair, 1, MPI_2INT, MPI_MAXLOC, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_min_pair, global_min_pair, 1, MPI_2INT, MPI_MINLOC, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Global MAX: %d from process %d\n", global_max_pair[0], global_max_pair[1]);
        printf("Global MIN: %d from process %d\n", global_min_pair[0], global_min_pair[1]);
    }

    MPI_Finalize();
    return 0;
}
