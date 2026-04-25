#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <cstdlib>

extern void correlate(int ny, int nx, const float* data, float* result);

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <ny> <nx>\n";
        return 1;
    }
    int ny = std::atoi(argv[1]);
    int nx = std::atoi(argv[2]);
    std::cout << "Running with ny=" << ny << ", nx=" << nx << "\n";

    std::vector<float> data(ny * nx);
    std::vector<float> result(ny * ny, 0.0f);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : data) v = dist(gen);

    auto start = std::chrono::high_resolution_clock::now();
    correlate(ny, nx, data.data(), result.data());
    auto end = std::chrono::high_resolution_clock::now();

    double time_taken = std::chrono::duration<double>(end - start).count();
    std::cout << "Time taken: " << std::fixed << std::setprecision(4) << time_taken << " seconds\n";

    // Print sample results
    std::cout << "\nSample correlations (first 5x5 upper triangle):\n";
    for (int i = 0; i < std::min(5, ny); ++i) {
        for (int j = 0; j <= i; ++j) {
            std::cout << std::setw(8) << std::setprecision(4) << result[i + j * ny] << " ";
        }
        std::cout << "\n";
    }
    return 0;
}
