#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

constexpr int N = 10000;
constexpr double EPS = 0.001;
constexpr double PI = 3.14159265358979323846;

inline int IND(int i, int j, int nx)
{
    return i * nx + j;
}

double wtime()
{
    using clock = std::chrono::high_resolution_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

int main()
{
    int nx = N, ny = N;
    std::vector<double> u(nx * ny, 0.0);
    std::vector<double> u_new(nx * ny, 0.0);
    double dx = 1.0 / (nx - 1.0);

    // Граничные условия
    for (int j = 0; j < nx; j++)
    {
        double x = dx * j;
        u[IND(0, j, nx)] = sin(PI * x);
        u[IND(ny - 1, j, nx)] = sin(PI * x) * exp(-PI);
        u_new[IND(0, j, nx)] = u[IND(0, j, nx)];
        u_new[IND(ny - 1, j, nx)] = u[IND(ny - 1, j, nx)];
    }

    double start = wtime();
    int iter = 0;

    while (true)
    {
        iter++;
        double maxdiff = 0;

        for (int i = 1; i < ny - 1; i++)
        {
            for (int j = 1; j < nx - 1; j++)
            {
                int idx = IND(i, j, nx);
                u_new[idx] = 0.25 * (u[IND(i - 1, j, nx)] +
                                     u[IND(i + 1, j, nx)] +
                                     u[IND(i, j - 1, nx)] +
                                     u[IND(i, j + 1, nx)]);
                maxdiff = std::max(maxdiff, std::abs(u_new[idx] - u[idx]));
            }
        }

        std::swap(u, u_new);
        if (maxdiff < EPS)
            break;
    }

    double end = wtime();
    std::cout << "Serial Laplace: N=" << N
              << ", iterations=" << iter
              << ", time=" << (end - start) << " s\n";

    return 0;
}
