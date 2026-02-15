#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

int main()
{
    int N = 500;
    int T = 500;

    vector<vector<double>> grid(N, vector<double>(N, 0.0));
    vector<vector<double>> new_grid(N, vector<double>(N, 0.0));

    // Initialize center heat
    grid[N/2][N/2] = 100.0;

    double start = omp_get_wtime();

    for(int t = 0; t < T; t++)
    {
        #pragma omp parallel for schedule(runtime)
        for(int i = 1; i < N-1; i++)
        {
            for(int j = 1; j < N-1; j++)
            {
                new_grid[i][j] = 0.25 * (
                    grid[i+1][j] +
                    grid[i-1][j] +
                    grid[i][j+1] +
                    grid[i][j-1]
                );
            }
        }

        grid.swap(new_grid);
    }

    double end = omp_get_wtime();

    cout << "Time: " << end - start << endl;

    return 0;
}
