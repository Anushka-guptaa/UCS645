#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_NUM 2000   // Change to larger value if needed (e.g. 10000)

int is_prime(int n) {
    if (n <= 1) return 0;
    if (n <= 3) return 1;
    if (n % 2 == 0 || n % 3 == 0) return 0;
    for (int i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) return 0;
    }
    return 1;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {  // MASTER
        int next_num = 2;
        int primes_found = 0;
        int* primes = (int*)malloc(MAX_NUM * sizeof(int));
        int active_slaves = size - 1;
        MPI_Status status;
        int msg;

        printf("Master: Finding primes up to %d using %d slaves...\n", MAX_NUM, active_slaves);

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
                primes[primes_found++] = msg;
            }
        }

        printf("Master: Found %d prime numbers:\n", primes_found);
        for (int i = 0; i < primes_found; i++) {
            printf("%d ", primes[i]);
            if ((i + 1) % 10 == 0) printf("\n");
        }
        printf("\n");
        free(primes);
    } 
    else {  // SLAVE
        int request = 0;
        MPI_Send(&request, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);  // initial request

        while (1) {
            int num;
            MPI_Recv(&num, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (num < 0) break;  // termination signal

            int result = is_prime(num) ? num : -num;
            MPI_Send(&result, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

            request = 0;
            MPI_Send(&request, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);  // request next number
        }
    }

    MPI_Finalize();
    return 0;
}
