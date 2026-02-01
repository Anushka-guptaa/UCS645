#include <stdio.h>
#include <omp.h>

#define STEPS 100000000

int main()
{
    double step = 1.0/STEPS;
    double sum = 0.0, pi;
    double start, end;

    start = omp_get_wtime();

    #pragma omp parallel for reduction(+:sum)
    for(int i=0;i<STEPS;i++)
    {
        double x = (i+0.5)*step;
        sum += 4.0/(1.0+x*x);
    }

    pi = step * sum;

    end = omp_get_wtime();

    printf("PI = %f\n", pi);
    printf("Time = %f\n", end-start);

    return 0;
}
