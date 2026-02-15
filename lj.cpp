#include <iostream>
#include <cmath>
#include <omp.h>
using namespace std;

int main()
{
    int N = 1000;
    double epsilon = 1.0;
    double sigma = 1.0;

    double *x = new double[N];
    double *y = new double[N];
    double *z = new double[N];

    double *fx = new double[N];
    double *fy = new double[N];
    double *fz = new double[N];

    for(int i = 0; i < N; i++)
    {
        x[i] = drand48();
        y[i] = drand48();
        z[i] = drand48();

        fx[i] = 0.0;
        fy[i] = 0.0;
        fz[i] = 0.0;
    }

    double total_energy = 0.0;

    double start = omp_get_wtime();

    for(int i = 0; i < N; i++)
    {
        for(int j = i+1; j < N; j++)
        {
            double dx = x[i] - x[j];
            double dy = y[i] - y[j];
            double dz = z[i] - z[j];

            double r2 = dx*dx + dy*dy + dz*dz;
            double inv_r2 = 1.0 / r2;
            double inv_r6 = inv_r2 * inv_r2 * inv_r2;
            double inv_r12 = inv_r6 * inv_r6;

            double force = 24 * epsilon * (2*inv_r12 - inv_r6) * inv_r2;

            fx[i] += force * dx;
            fy[i] += force * dy;
            fz[i] += force * dz;

            fx[j] -= force * dx;
            fy[j] -= force * dy;
            fz[j] -= force * dz;

            total_energy += 4 * epsilon * (inv_r12 - inv_r6);
        }
    }

    double end = omp_get_wtime();

    cout << "Energy: " << total_energy << endl;
    cout << "Time: " << end - start << endl;

    delete[] x;
    delete[] y;
    delete[] z;
    delete[] fx;
    delete[] fy;
    delete[] fz;

    return 0;
}
