#include <iostream>
#include <vector>
#include <cmath>
#include <mpi.h>

constexpr double EPS = 0.001;
constexpr double PI = 3.14159265358979323846;

inline int IND(int i, int j, int nx)
{
    return i * (nx + 2) + j;
}

int get_block_size(int n, int rank, int nprocs)
{
    int s = n / nprocs;
    if (n % nprocs > rank)
        s++;
    return s;
}

int get_sum_of_prev_blocks(int n, int rank, int nprocs)
{
    int rem = n % nprocs;
    return (n / nprocs) * rank + ((rank >= rem) ? rem : rank);
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int commsize, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &commsize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double ttotal = -MPI_Wtime();

    MPI_Comm cartcomm;
    int dims[2] = {0, 0}, periodic[2] = {0, 0};
    MPI_Dims_create(commsize, 2, dims);
    int px = dims[0], py = dims[1];

    if (px < 2 || py < 2)
    {
        if (rank == 0)
            std::cerr << "Invalid process topology: px and py must be >= 2\n";
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periodic, 0, &cartcomm);

    int coords[2];
    MPI_Cart_coords(cartcomm, rank, 2, coords);
    int rankx = coords[0], ranky = coords[1];

    int rows, cols;

    if (rank == 0)
    {
        rows = (argc > 1) ? std::atoi(argv[1]) : py * 100;
        cols = (argc > 2) ? std::atoi(argv[2]) : px * 100;
        if (rows < py || cols < px)
        {
            std::cerr << "Rows or columns less than process grid\n";
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        int args[2] = {rows, cols};
        MPI_Bcast(&args, 2, MPI_INT, 0, MPI_COMM_WORLD);
    }
    else
    {
        int args[2];
        MPI_Bcast(&args, 2, MPI_INT, 0, MPI_COMM_WORLD);
        rows = args[0];
        cols = args[1];
    }

    int ny = get_block_size(rows, ranky, py);
    int nx = get_block_size(cols, rankx, px);

    std::vector<double> local_grid((ny + 2) * (nx + 2), 0.0);
    std::vector<double> local_newgrid((ny + 2) * (nx + 2), 0.0);

    double dx = 1.0 / (cols - 1.0);
    int sj = get_sum_of_prev_blocks(cols, rankx, px);

    if (ranky == 0)
    {
        for (int j = 1; j <= nx; j++)
        {
            double x = dx * (sj + j - 1);
            int ind = IND(0, j, nx);
            local_grid[ind] = local_newgrid[ind] = std::sin(PI * x);
        }
    }

    if (ranky == py - 1)
    {
        for (int j = 1; j <= nx; j++)
        {
            double x = dx * (sj + j - 1);
            int ind = IND(ny + 1, j, nx);
            local_grid[ind] = local_newgrid[ind] = std::sin(PI * x) * std::exp(-PI);
        }
    }

    int left, right, top, bottom;
    MPI_Cart_shift(cartcomm, 0, 1, &left, &right);
    MPI_Cart_shift(cartcomm, 1, 1, &top, &bottom);

    MPI_Datatype col, row;
    MPI_Type_vector(ny, 1, nx + 2, MPI_DOUBLE, &col);
    MPI_Type_commit(&col);

    MPI_Type_contiguous(nx, MPI_DOUBLE, &row);
    MPI_Type_commit(&row);

    MPI_Request reqs[8];
    double thalo = 0, treduce = 0;
    int niters = 0;

    for (;;)
    {
        niters++;
        for (int i = 1; i <= ny; i++)
            for (int j = 1; j <= nx; j++)
                local_newgrid[IND(i, j, nx)] =
                    0.25 * (local_grid[IND(i - 1, j, nx)] +
                            local_grid[IND(i + 1, j, nx)] +
                            local_grid[IND(i, j - 1, nx)] +
                            local_grid[IND(i, j + 1, nx)]);

        double maxdiff = 0;
        for (int i = 1; i <= ny; i++)
            for (int j = 1; j <= nx; j++)
                maxdiff = std::max(maxdiff,
                                   std::abs(local_newgrid[IND(i, j, nx)] - local_grid[IND(i, j, nx)]));

        std::swap(local_grid, local_newgrid);

        treduce -= MPI_Wtime();
        MPI_Allreduce(MPI_IN_PLACE, &maxdiff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        treduce += MPI_Wtime();

        if (maxdiff < EPS)
            break;

        thalo -= MPI_Wtime();
        MPI_Irecv(&local_grid[IND(0, 1, nx)], 1, row, top, 0, cartcomm, &reqs[0]);
        MPI_Irecv(&local_grid[IND(ny + 1, 1, nx)], 1, row, bottom, 0, cartcomm, &reqs[1]);
        MPI_Irecv(&local_grid[IND(1, 0, nx)], 1, col, left, 0, cartcomm, &reqs[2]);
        MPI_Irecv(&local_grid[IND(1, nx + 1, nx)], 1, col, right, 0, cartcomm, &reqs[3]);

        MPI_Isend(&local_grid[IND(1, 1, nx)], 1, row, top, 0, cartcomm, &reqs[4]);
        MPI_Isend(&local_grid[IND(ny, 1, nx)], 1, row, bottom, 0, cartcomm, &reqs[5]);
        MPI_Isend(&local_grid[IND(1, 1, nx)], 1, col, left, 0, cartcomm, &reqs[6]);
        MPI_Isend(&local_grid[IND(1, nx, nx)], 1, col, right, 0, cartcomm, &reqs[7]);

        MPI_Waitall(8, reqs, MPI_STATUS_IGNORE);
        thalo += MPI_Wtime();
    }

    MPI_Type_free(&row);
    MPI_Type_free(&col);

    ttotal += MPI_Wtime();

    if (rank == 0)
    {
        std::cout << "# Heat 2D (mpi): grid: rows " << rows << ", cols " << cols
                  << ", procs " << commsize << " (px " << px << ", py " << py << ")\n"
                  << "# total time = " << ttotal << " s\n";

        double T_serial = 48.640;
        double speedup = T_serial / ttotal;
        std::cout << "# speedup = " << speedup << '\n';
    }

    MPI_Finalize();
    return 0;
}
