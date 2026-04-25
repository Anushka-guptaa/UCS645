#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_NUM 10000   // Change to larger value if needed

int is_perfect(int n) {
    if (n <= 1) return 0;
    int sum = 1;
    for (int i = 2; i * i <= n; i++) {
        if (n % i == 0) {
            sum += i;
            if (i != n / i) sum += n / i;
        }
    }
    return (sum == n);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {  // MASTER
        int next_num = 2;
        int perfect_found = 0;
        int* perfects = (int*)malloc(100 * sizeof(int));  // enough space
        int active_slaves = size - 1;
        MPI_Status status;
        int msg;

        printf("Master: Finding perfect numbers up to %d using %d slaves...\n", MAX_NUM, active_slaves);

        while (active_slaves > 0) {
            MPI_Recv(&msg, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG,
                     MPI_COMM_WORLD, &status);
            int src = status.MPI_SOURCE;

            if (msg == 0) {  // Slave requesting work
                if (next_num <= MAX_NUM) {
                    MPI_Send(&next_num, 1, MPI_INT, src, 0, MPI_COMM_WORLD);
                    next_num++;
                } else {
                    int terminate = -1;
                    MPI_Send(&terminate, 1, MPI_INT, src, 0, MPI_COMM_WORLD);
                    active_slaves--;
                }
            } else if (msg > 0) {
                perfects[perfect_found++] = msg;
            }
        }

        printf("Master: Found %d perfect numbers: ", perfect_found);
        for (int i = 0; i < perfect_found; i++) {
            printf("%d ", perfects[i]);
        }
        printf("\n");
        free(perfects);
    } 
    else {  // SLAVE
        int request = 0;
        MPI_Send(&request, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);  // initial request

        while (1) {
            int num;
            MPI_Recv(&num, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (num < 0) break;  // termination

            int result = is_perfect(num) ? num : -num;
            MPI_Send(&result, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

            request = 0;
            MPI_Send(&request, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);  // request next
        }
    }

    MPI_Finalize();
    return 0;
}
