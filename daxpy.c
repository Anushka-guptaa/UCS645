#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 50000000

int main()
{
    double *X, *Y, a = 2.5;
    double start, end;

    X = (double*)malloc(N * sizeof(double));
    Y = (double*)malloc(N * sizeof(double));

    for(int i=0;i<N;i++)
    {
        X[i] = i;
        Y[i] = i;
    }

    start = omp_get_wtime();

    #pragma omp parallel for
    for(int i=0;i<N;i++)
    {
        X[i] = a*X[i] + Y[i];
    }

    end = omp_get_wtime();

    printf("Time: %f seconds\n", end-start);

    return 0;
}
