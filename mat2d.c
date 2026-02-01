#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 600

int main()
{
    static double A[N][N], B[N][N], C[N][N];
    double start, end;

    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++)
        {
            A[i][j]=1;
            B[i][j]=1;
            C[i][j]=0;
        }

    start = omp_get_wtime();

    #pragma omp parallel for collapse(2)
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++)
            for(int k=0;k<N;k++)
                C[i][j]+=A[i][k]*B[k][j];

    end = omp_get_wtime();

    printf("Time: %f\n", end-start);

    return 0;
}
