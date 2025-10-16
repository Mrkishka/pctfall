#include <mpi.h>
#include <iostream>
#include <cmath>
#include <cstdlib>

#define N 10000000

double func(double x, double y)
{
    return std::exp((x + y) * (x + y));
}

double getrand()
{
    return static_cast<double>(rand()) / RAND_MAX;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, commsize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &commsize);

    srand(rank + 1);

    const int n = N;
    double local_sum = 0.0;
    int local_count = 0;

    double start_time = 0.0;
    if (rank == 0)
        start_time = MPI_Wtime();

    for (int i = rank; i < n; i += commsize)
    {
        double x = getrand();
        double y = getrand() * (1.0 - x);

        local_sum += func(x, y);
        local_count++;
    }

    double global_sum = 0.0;
    int global_count = 0;

    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        double area = 0.5;
        double mean_value = global_sum / global_count;
        double result = mean_value * area;

        double finish_time = MPI_Wtime();
        double elapsed = finish_time - start_time;

        std::cout << "Time: " << elapsed
                  << " sec\nResult: " << result
                  << "\nN = " << n << std::endl;

        double base_time = 0.616065;
        double speedup = base_time / elapsed;
        std::cout << "Speedup: " << speedup << std::endl;
    }

    MPI_Finalize();
    return 0;
}
