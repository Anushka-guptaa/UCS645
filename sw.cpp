#include <iostream>
#include <vector>
#include <algorithm>
#include <omp.h>
using namespace std;

int main()
{
    string s1 = "ACACACTA";
    string s2 = "AGCACACA";

    int m = s1.length();
    int n = s2.length();

    int match = 2;
    int mismatch = -1;
    int gap = -1;

    vector<vector<int>> H(m+1, vector<int>(n+1, 0));

    int max_score = 0;

    double start = omp_get_wtime();

    for(int i = 1; i <= m; i++)
    {
        for(int j = 1; j <= n; j++)
        {
            int score_diag = H[i-1][j-1] + (s1[i-1] == s2[j-1] ? match : mismatch);
            int score_up = H[i-1][j] + gap;
            int score_left = H[i][j-1] + gap;

            H[i][j] = max({0, score_diag, score_up, score_left});

            max_score = max(max_score, H[i][j]);
        }
    }

    double end = omp_get_wtime();

    cout << "Max Score: " << max_score << endl;
    cout << "Time: " << end - start << endl;

    return 0;
}
