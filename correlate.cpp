#include <cmath>
#include <vector>
#include <omp.h>

void correlate(int ny, int nx, const float* data, float* result) {
    std::vector<double> row_sum(ny, 0.0);
    std::vector<double> row_sum_sq(ny, 0.0);

    // Parallel precomputation
    #pragma omp parallel for
    for (int i = 0; i < ny; ++i) {
        double s = 0.0, sq = 0.0;
        for (int k = 0; k < nx; ++k) {
            double val = data[i * nx + k];
            s += val;
            sq += val * val;
        }
        row_sum[i] = s;
        row_sum_sq[i] = sq;
    }

    // Parallel correlation (j <= i)
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j <= i; ++j) {
            double dot = 0.0;
            for (int k = 0; k < nx; ++k) {
                dot += static_cast<double>(data[i*nx + k]) * data[j*nx + k];
            }
            double num = static_cast<double>(nx) * dot - row_sum[i] * row_sum[j];
            double den1 = static_cast<double>(nx) * row_sum_sq[i] - row_sum[i]*row_sum[i];
            double den2 = static_cast<double>(nx) * row_sum_sq[j] - row_sum[j]*row_sum[j];
            double corr = 0.0;
            if (den1 > 0.0 && den2 > 0.0) {
                corr = num / std::sqrt(den1 * den2);
            }
            result[i + j * ny] = static_cast<float>(corr);
        }
    }
}
