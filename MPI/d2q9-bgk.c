/*
On second thoughts, maybe split by sqrt(size) x sqrt(size)?
For a total number of cells a = hw and a split between n = yx cores,
    the number of cells per core is a/n = hw/(yx),
    the number of non-edge cells per core is (y-2)(x-2) = n - 2y - 2x + 4,
    the fraction of non-edge cells per core is (n - 2y - 2n/y + 4) / (a/n) = n/a (n - 2y - 2n/y + 4),
    maximising this fraction:
        d/dy n/a (n - 2y - 2n/y + 4) = 2n/a (n/y² - 1)
        2n/a (n/y² - 1) = 0 -> y = sqrt(n)
        d/dy 2n/a (n/y² - 1) = -4n²/(ay³)
        -4n²/(a sqrt³(n)) = -4 sqrt(n)/a, guaranteed to be non-positive, so a local maximum
My intuition was spot on!

Initialisation, cs = 0. Modifying cells cs, referencing cells 1 - cs
   __________________________________________ ______________________________________________________________________________________________________________________
  / \          \         \         \         \    Node 1 will receive node 2 [cs, top]. Node 1 processes [1-cs, other rows] from top to bottom.
 /   \  Core 0  \ Core 1  \ Core 2  \ Core 3  \   When node 1 receives from node 2, it calculates [1-cs, bottom] and sends to node 2.
 |   |__________|_________|_________|_________|__ cs = 1-cs when done.______________________________________________________________________________________________
 /   \          \         \         \         \   Node 2 will receive node 1 [cs, bottom], node 3 [cs, top]. Node 2 processes [1-cs, other rows] from top to bottom.
|  T  | Core 4   | Core 5  | Core 6  | Core 7  |  When node 2 receives from node 1, it calculates [1-cs, top] and sends to node 1.
/  o  \          \         \         \         \  When node 2 receives from node 3, it calculates [1-cs, bottom] and sends to node 3.
|  r  |__________|_________|_________|_________|_ cs = 1-cs when done.______________________________________________________________________________________________
|  u  |          |         |         |         |  Node 3 will receive node 2 [cs, bottom], node 4 [cs, top]. Node 3 processes [1-cs, other rows] from top to bottom.
\  s  /          /         /         /         /  When node 3 receives from node 2, it calculates [1-cs, top] and sends to node 2.
|     | Core 8   | Core 9  | Core 10 | Core 11 |  When node 3 receives from node 4, it calculates [1-cs, bottom] and sends to node 4.
 \   /__________/_________/_________/_________/__ cs = 1-cs when done.______________________________________________________________________________________________
 |   |          |         |         |         |   Node 4 will receive node 3 [cs, bottom]. Node 4 processes [1-cs, other rows] from top to bottom.
 \   / Core 12  / Core 13 / Core 14 / Core 15 /   When node 4 receives from node 3, it calculates [1-cs, top] and sends to node 3.
  \ /          /         /         /         /    cs = 1-cs when done.
   ¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯
Finalisation, reduce+ tot_u


Initialisation:
    Communication is constructed between pairs of adjacent cores (16 in total).
    Each node gathers their parameters and allocate their memory.
    Each node reads their quarter of the obstacles and generates their grids.

*/

// local_density is oscillating between two values

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cmath>
#include <mpi.h>
#ifdef __GNUC__
    #include <sys/time.h>
    #include <sys/resource.h>
#else
    #include <chrono>
#endif

//#define DEBUG
//#define TIMING

const char* const AVVELSFILE     = "av_vels.dat";
const char* const FINALSTATEFILE = "final_state.dat";

enum
{
    CR,
    TR,
    BR,
    CL,
    TL,
    BL,
    CC,
    TC,
    BC,
    NSPEEDS
};

template <typename T>
struct Matrix
{
    Matrix(const int rows_in, const int columns_in)
        : rows(rows_in), columns(columns_in)
    {
        data = new(std::nothrow) T[rows_in * columns];
    }

    ~Matrix()
    {
        delete[] data;
    }

    T* operator[](const int row) const
    {
        return &data[row * columns];
    }

    bool operator!()
    {
        return data == NULL;
    }

    T* data;
    int rows, columns;
};

struct t_param
{
    int nx,           // no. of cells in x-direction
        ny,           // no. of cells in y-direction
        maxIters,     // no. of iterations
        reynolds_dim; // dimension for Reynolds number
    double density,   // density per link
           accel,     // density redistribution
           omega;     // relaxation parameter
};

struct t_speed // Almost certainly going to have to turn this into an struct of arrays (I know what I mean)
{
    double speeds[NSPEEDS];
};


//____________________________________________________________________________________________________//


double PropagateReboundCollision(int y, int x, Matrix<t_speed>& cells, Matrix<t_speed>& cells_temp, const Matrix<int>& obstacles, const t_param& parameters)
{
    double* const& speeds = cells_temp[y][x].speeds;
    double tot_u(0);

    // Propagate //
    {
        // These should now be unnecessary
        const int y_n = y != cells.rows - 1    ? y + 1 : 0;
        const int x_e = x != cells.columns - 1 ? x + 1 : 0;
        const int y_s = y > 0                  ? y - 1 : cells.rows - 1;
        const int x_w = x > 0                  ? x - 1 : cells.columns - 1;

        // Propagate densities to neighbouring cells, following appropriate directions of travel and writing into scratch space grid
        speeds[CR] = cells[y]  [x_w].speeds[CR]; // Centre right
        speeds[TR] = cells[y_s][x_w].speeds[TR]; // Top    right
        speeds[BR] = cells[y_n][x_w].speeds[BR]; // Bottom right
        speeds[CL] = cells[y]  [x_e].speeds[CL]; // Centre left
        speeds[TL] = cells[y_s][x_e].speeds[TL]; // Top    left
        speeds[BL] = cells[y_n][x_e].speeds[BL]; // Bottom left
        speeds[CC] = cells[y]  [x]  .speeds[CC]; // Centre
        speeds[TC] = cells[y_s][x]  .speeds[TC]; // Top    centre
        speeds[BC] = cells[y_n][x]  .speeds[BC]; // Bottom centre
    }
    
    // Collision //
    {
        // Rebound //
        if (obstacles[y-1][x])
        {
            std::swap(speeds[CR], speeds[CL]); // Centre right  <-> Centre left
            std::swap(speeds[TC], speeds[BC]); // Top    centre <-> Bottom centre
            std::swap(speeds[TR], speeds[BL]); // Top    right  <-> Bottom left
            std::swap(speeds[TL], speeds[BR]); // Top    left   <-> Bottom right
            return 0.0;
        }

        // compute local density total
        double local_density(0.0);

        for (int iii(0); iii < NSPEEDS; ++iii)
            local_density += speeds[iii];

        // compute x velocity component
        double u_x =
            (
                  speeds[CR]
                + speeds[TR]
                + speeds[BR]
                - speeds[CL]
                - speeds[TL]
                - speeds[BL]
            )
            / local_density;

        // compute y velocity component
        double u_y =
            (
                  speeds[TC]
                + speeds[TR]
                + speeds[TL]
                - speeds[BC]
                - speeds[BL]
                - speeds[BR]
            )
            / local_density;

        // Equilibrium densities
        double d_equ[NSPEEDS];
        
        {
            double x, y, z, w;
            const double
                w0(4.0 / 9.0 / 2.0),
                w1(1.0 / 9.0 / 2.0),
                w2(1.0 / 36.0 / 2.0),
                a = 3.0,
                b = 1.0 - (u_x*u_x + u_y*u_y)*a;

                               d_equ[CC] = (1.0 + b) * w0;
            x = u_x * a + 1.0; d_equ[CR] = (x*x + b) * w1;
            y = u_y * a + 1.0; d_equ[TC] = (y*y + b) * w1;
            z = 2.0 - x;       d_equ[CL] = (z*z + b) * w1;
            w = 2.0 - y;       d_equ[BC] = (w*w + b) * w1;
            z = x + y - 1.0;   d_equ[TR] = (z*z + b) * w2;
            w = y - x + 1.0;   d_equ[TL] = (w*w + b) * w2;
            z = 2.0 - z;       d_equ[BL] = (z*z + b) * w2;
            w = 2.0 - w;       d_equ[BR] = (w*w + b) * w2;
        }
        

        // Relaxation
        for (int iii(0); iii < NSPEEDS; ++iii)
            speeds[iii] = speeds[iii] + parameters.omega*(d_equ[iii] * local_density - speeds[iii]);


        // Average velocity //
        local_density = 0.0;

        for (int iii(0); iii < NSPEEDS; ++iii)
            local_density += speeds[iii];

        u_x =
            (
                  speeds[CR]
                + speeds[TR]
                + speeds[BR]
                - speeds[CL]
                - speeds[TL]
                - speeds[BL]
            );

        u_y =
            (
                  speeds[TC]
                + speeds[TR]
                + speeds[TL]
                - speeds[BC]
                - speeds[BL]
                - speeds[BR]
            );

        // Accumulate the norm of velocity
        tot_u += std::sqrt(u_x*u_x + u_y*u_y) / local_density;
    }
    
    return tot_u;
}


double calc_av(const MPI::Intracomm& comm, double average_velocity, const int rank, const int first, const int last, const int tot_cells)
{
    if (rank == first)
        for (int i(last); i; --i)
        {
            double average_velocity_foreign;
            comm.Recv(&average_velocity_foreign, 1, MPI::DOUBLE, MPI::ANY_SOURCE, 2);
            average_velocity += average_velocity_foreign;
        }
    else
        comm.Send(&average_velocity, 1, MPI::DOUBLE, first, 2);

    return average_velocity / double(tot_cells);
}


double total_density(const MPI::Intracomm& comm, const Matrix<t_speed>& cells, const int rank, const int first, const int last)
{
    double total = 0.0;

    for (int y(1); y < cells.rows - 1; ++y)
        for (int x(0); x < cells.columns; ++x)
            for (int t(0); t < NSPEEDS; ++t)
                total += cells[y][x].speeds[t];

    if (rank == first)
        for (int i(last); i; --i)
        {
            double total_foreign;
            comm.Recv(&total_foreign, 1, MPI::DOUBLE, MPI::ANY_SOURCE, 2);
            total += total_foreign;
        }
    else
        comm.Send(&total, 1, MPI::DOUBLE, first, 2);

    return total;
}


void write_values(const MPI::Intracomm& comm, const t_param& params, const Matrix<t_speed>& cells, const Matrix<int>& obstacles, const double* av_vels, const int rank, const int size, const int first)
{
    {
        std::ofstream file_state_file(FINALSTATEFILE, std::ios::out | std::ios::trunc);
        if (!file_state_file)
        {
            std::cerr << "Could not open output final state file " << FINALSTATEFILE << '\n';
            std::exit(EXIT_FAILURE);
        }
        comm.Barrier();

        file_state_file.setf(std::ios_base::scientific, std::ios_base::floatfield);
        file_state_file.setf(std::ios_base::uppercase);
        file_state_file.precision(12);

        for (int i(0); i < size; ++i)
        {
            if (rank == i)
                for (int y(0); y < cells.rows - 2; ++y)
                    for (int x(0); x < cells.columns; ++x)
                    {
                        const double c_sq = 1.0 / 3.0; // sq. of speed of sound
                        double
                            pressure,               // fluid pressure in grid cell
                            u_x,                    // x-component of velocity in grid cell
                            u_y,                    // y-component of velocity in grid cell
                            u;                      // norm--root of summed squares--of u_x and u_y

                        // An occupied cell
                        if (obstacles[y][x])
                        {
                            u_x = u_y = u = 0.0;
                            pressure = params.density * c_sq;
                        }
                        // No obstacle
                        else
                        {
                            // Compute local density total
                            double local_density = 0.0;

                            for (int i(0); i < NSPEEDS; ++i)
                                local_density += cells[y+1][x].speeds[i];

                            // Compute x velocity component
                            u_x =
                                (
                                      cells[y+1][x].speeds[CR]
                                    + cells[y+1][x].speeds[TR]
                                    + cells[y+1][x].speeds[BR]
                                    - cells[y+1][x].speeds[CL]
                                    - cells[y+1][x].speeds[TL]
                                    - cells[y+1][x].speeds[BL]
                                )
                                / local_density;

                            // Compute y velocity component
                            u_y =
                                (
                                      cells[y+1][x].speeds[TC]
                                    + cells[y+1][x].speeds[TR]
                                    + cells[y+1][x].speeds[TL]
                                    - cells[y+1][x].speeds[BC]
                                    - cells[y+1][x].speeds[BL]
                                    - cells[y+1][x].speeds[BR]
                                )
                                / local_density;

                            // Compute norm of velocity
                            u = sqrt(u_x*u_x + u_y*u_y);

                            // Compute pressure
                            pressure = local_density * c_sq;
                        }

                        // write to file
                        file_state_file << x << ' ' << y << ' ' << u_x << ' ' << u_y << ' ' << u << ' ' << pressure << ' ' << obstacles[y][x] << '\n';
                    }
            file_state_file.flush();
            comm.Barrier();
            file_state_file.seekp(0, std::ios_base::end);
        }
    }

    if (rank == first)
    {
        std::ofstream average_values_file(AVVELSFILE);
        if (!average_values_file)
        {
            std::cerr << "Could not open output average values file " << AVVELSFILE << '\n';
            std::exit(EXIT_FAILURE);
        }

        average_values_file.setf(std::ios_base::scientific, std::ios_base::floatfield);
        average_values_file.setf(std::ios_base::uppercase);
        average_values_file.precision(12);

        for (int i(0); i < params.maxIters; ++i)
            average_values_file << i << ":\t" << av_vels[i] << '\n';
    }
}


int main(int argc, char* argv[])
{
    // Initialisation //
    MPI::Init(argc, argv);
    
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <paramfile> <obstaclefile>\n";
        MPI::Finalize();
        return EXIT_FAILURE;
    }

    char*& parameters_filepath = argv[1];
    char*& obstacles_filepath = argv[2];

    std::cout.sync_with_stdio(false);
    std::cerr.sync_with_stdio(false);
    std::cout.setf(std::ios_base::scientific, std::ios_base::floatfield);
    std::cout.setf(std::ios_base::uppercase);
    std::cout.precision(12);

    // Process parameter file
    t_param parameters;
    {
        MPI::File parameters_file(MPI::File::Open(MPI::COMM_WORLD, parameters_filepath, MPI::MODE_RDONLY, MPI::INFO_NULL));

        if (!parameters_file)
        {
            std::cerr << "Could not open input parameter file " << parameters_filepath << '\n';
            parameters_file.Close();
            MPI::Finalize();
            return EXIT_FAILURE;
        }
        
        std::string parameters_string(parameters_file.Get_size(), char());
        parameters_file.Read_all((void*)(parameters_string.c_str()), parameters_string.length(), MPI::CHAR);
        parameters_file.Close();
        std::istringstream parameters_stream(parameters_string);

        // Read in the parameter values
        parameters_stream >> parameters.nx >> parameters.ny >> parameters.maxIters >> parameters.reynolds_dim >> parameters.density >> parameters.accel >> parameters.omega;
        if (!parameters_stream)
        {
            std::cerr << "Invalid parameters in input parameter file\n";
            MPI::Finalize();
            return EXIT_FAILURE;
        }
    }

    // Main grid
#ifdef TIMING
    std::ofstream timing("timing.txt", std::ios::out | std::ios::trunc);
    for (int size(64); size; --size)
    {
#else
    const int size(MPI::COMM_WORLD.Get_size());
#endif
        const int& nx(parameters.nx);
        const int
            rank(MPI::COMM_WORLD.Get_rank()),
            last(size - 1),
            first(0),
            division(parameters.ny/size),
            remainder(parameters.ny - division*size),
            spill(rank < remainder),
            ny(division + spill),
            end(spill ? ny*(rank + 1) : ny*(rank + 1 - remainder) + remainder*(division + 1)),
            start(end - ny);

        const int ranges[3] = {0, size - 1, 1};
        MPI::Group group(MPI::COMM_WORLD.Get_group().Range_incl(1, &ranges));
        MPI::Intracomm comm(MPI::COMM_WORLD.Create(group));
        if (comm == MPI::COMM_NULL)
        {
        #ifdef TIMING
            continue;
        #else
            MPI::Finalize();
            return EXIT_SUCCESS;
        #endif
        }

        Matrix<t_speed> cells_all[2] = {{ny + 2, nx}, {ny + 2, nx}};
        Matrix<t_speed>& cells = cells_all[0];
        if (!cells)
        {
            std::cerr << "Cannot allocate memory for cells\n";
            MPI::Finalize();
            return EXIT_FAILURE;
        }

        double
            density_centre   = parameters.density * 4.0 / 9.0,
            density_axis     = density_centre / 4.0,
            density_diagonal = density_axis / 4.0;

        for (int y = 0; y < cells.rows; ++y)
            for (int x = 0; x < cells.columns; ++x)
            {
                cells[y][x].speeds[CR] = density_axis;
                cells[y][x].speeds[TR] = density_diagonal;
                cells[y][x].speeds[BR] = density_diagonal;
                cells[y][x].speeds[CL] = density_axis;
                cells[y][x].speeds[TL] = density_diagonal;
                cells[y][x].speeds[BL] = density_diagonal;
                cells[y][x].speeds[CC] = density_centre;
                cells[y][x].speeds[TC] = density_axis;
                cells[y][x].speeds[BC] = density_axis;
            }

        // 'Helper' grid, used as scratch space
        Matrix<t_speed>& cells_temp = cells_all[1];
        if (!cells_temp)
        {
            std::cerr << "Cannot allocate memory for temporary cells\n";
            MPI::Finalize();
            return EXIT_FAILURE;
        }

        // The map of obstacles
        Matrix<int> obstacles(ny, nx); // Change this to bool
        if (!obstacles)
        {
            std::cerr << "Cannot allocate memory for obstacles\n";
            MPI::Finalize();
            return EXIT_FAILURE;
        }

        // First set all cells in obstacle array to zero
        std::memset(obstacles.data, 0, obstacles.columns * obstacles.rows * sizeof(*obstacles.data));

        int tot_cells(0);
        // Process obstacle data file
        {
            MPI::File obstacle_file(MPI::File::Open(comm, obstacles_filepath, MPI::MODE_RDONLY, MPI::INFO_NULL));
            if (!obstacle_file)
            {
                std::cerr << "Could not open input obstacle file " << obstacles_filepath << '\n';
                obstacle_file.Close();
                MPI::Finalize();
                return EXIT_FAILURE;
            }
        
            std::string obstacles_string(obstacle_file.Get_size(), char());
            obstacle_file.Read_all((void*)(obstacles_string.c_str()), obstacles_string.length(), MPI_CHAR);
            obstacle_file.Close();
            std::istringstream obstacles_stream(obstacles_string);

            for (;;)
            {
                int x, y, blocked;
                obstacles_stream >> x >> y >> blocked;
                if (!obstacles_stream)
                {
                    if (obstacles_stream.eof())
                        break;
                    std::cerr << "Invalid obstacle in input obstacle file\n";
                    MPI::Finalize();
                    return EXIT_FAILURE;
                }

                if (x >= parameters.nx)
                {
                    std::cerr << "Invalid obstacle x co-ordinate in input obstacle file\n";
                    MPI::Finalize();
                    return EXIT_FAILURE;
                }

                if (y >= parameters.ny)
                {
                    std::cerr << "Invalid obstacle y co-ordinate in input obstacle file\n";
                    MPI::Finalize();
                    return EXIT_FAILURE;
                }

                if (blocked != 1)
                {
                    std::cerr << "Invalid obstacle blocked value in input obstacle file. Required to be 1 - why am I even reading this value?!\n";
                    MPI::Finalize();
                    return EXIT_FAILURE;
                }

                if (unsigned(x - 0) >= unsigned(obstacles.columns))
                    continue;
                if (unsigned(y - start) >= unsigned(obstacles.rows)) // y - start < 0 || y - start >= rows
                    continue;

                obstacles[y-start][x] = 1;
                ++tot_cells;
            }
        }
        if (rank == first)
            for (int i(last); i; --i)
            {
                int tot_cells_foreign(0);
                comm.Recv(&tot_cells_foreign, 1, MPI::INT, MPI::ANY_SOURCE, 2);
                tot_cells += tot_cells_foreign;
            }
        else
            comm.Send(&tot_cells, 1, MPI::INT, first, 2);
        tot_cells = parameters.ny * parameters.nx - tot_cells;
        comm.Bcast(&tot_cells, 1, MPI::INT, first);

        // Allocate space to hold a record of the average velocities computed at each time step
        // Perhaps RMA-accumulate this into the rank=first
        double* average_velocity(new double[parameters.maxIters]);
    
        // MPI message buffer
        // t_speed* message_buffer(new t_speed[cells.columns * 2]); // Plus some overhead, see http://www.mpi-forum.org/docs/mpi-2.1/mpi21-report-bw/node54.htm

        double tot_u(0.0);
        int cells_switch(0);
    #ifdef __GNUC__
        struct timeval timstr;
        double tic, toc, usrtim, systim;
        struct rusage ru;
    #else
        std::chrono::system_clock::time_point clock_start, clock_end;
    #endif

        // Main loop //
        {
        #ifdef DEBUG
            if (rank == first)
            {
                std::cout << "Waiting..." << std::endl;
                std::string debug_s;
                std::cin >> debug_s;
            }
        #endif

            comm.Barrier();
            if (rank == first)
            {
            #ifdef __GNUC__
                gettimeofday(&timstr, NULL);
                tic = timstr.tv_sec + timstr.tv_usec / 1000000.0;
            #else
                clock_start = std::chrono::high_resolution_clock::now();
            #endif
            }

            for (int i(0); i < parameters.maxIters; ++i)
            {
                Matrix<t_speed>& cells      = cells_all[cells_switch];
                Matrix<t_speed>& cells_temp = cells_all[1-cells_switch];
                tot_u = 0.0;
            
                // Request edge cases straight away
                MPI::Request
                    fromAbove(comm.Irecv(cells[0],              cells.columns * NSPEEDS, MPI::DOUBLE, (rank + last) % size, cells_switch)),
                    fromBelow(comm.Irecv(cells[cells.rows - 1], cells.columns * NSPEEDS, MPI::DOUBLE, (rank + 1)    % size, cells_switch));
                bool receivedAbove(false), receivedBelow(false);
            
                // Accelerate flow //
                if (rank == last)
                {
                    // Weights
                    const double
                        w1(parameters.density * parameters.accel / 9.0), // Speed centre row
                        w2(w1 / 4.0);                                    // Speed top/bottom rows

                    for (int x = 0; x < cells.columns; ++x)
                        if (!obstacles[obstacles.rows - 2][x])
                        {
                            double* const& speeds = cells[cells.rows - 3][x].speeds;
                            if
                            (
                                   speeds[CL] - w1 > 0.0 // Centre left
                                && speeds[TL] - w2 > 0.0 // Top left
                                && speeds[BL] - w2 > 0.0 // Bottom left
                            )
                            {
                                speeds[CL] -= w1; // Centre left
                                speeds[TL] -= w2; // Top left
                                speeds[BL] -= w2; // Bottom left
                                speeds[CR] += w1; // Centre right
                                speeds[TR] += w2; // Top right
                                speeds[BR] += w2; // Bottom right
                            }
                    }
                }
            
                // Send edge cases, to be used as reference for neighbour.
                // Some research suggests the system buffer should be sufficiently large
                comm.Isend(cells[1],              cells.columns * NSPEEDS, MPI::DOUBLE, (rank + last) % size, cells_switch);
                comm.Isend(cells[cells.rows - 2], cells.columns * NSPEEDS, MPI::DOUBLE, (rank + 1)    % size, cells_switch);
            
                if (!receivedAbove)
                {
                    fromAbove.Wait();
                    receivedAbove = true;
                }
                if (!receivedBelow)
                {
                    fromBelow.Wait();
                    receivedBelow = true;
                }
                for (int y(1); y < cells.rows - 1; ++y) for (int x(0); x < cells.columns; ++x)
                {
                    tot_u += PropagateReboundCollision(y, x, cells, cells_temp, obstacles, parameters);
                }

                average_velocity[i] = tot_u;

            #ifdef DEBUG
                if (
                    !(i % 1000)
                    //i < 100
                    //&& false
                )
                {
                    double
                        debug_av(calc_av(comm, average_velocity[i], rank, first, last, tot_cells)),
                        debug_td(total_density(comm, cells, rank, first, last));
                    if (rank == first)
                        std::cout
                            << "==timestep: " << i << '/' << parameters.maxIters << "==\n"
                            << "av velocity: " << debug_av << '\n'
                            << "tot density: " << debug_td << std::endl;
                }
            #endif
                cells_switch = 1 - cells_switch;
            }
        
            // Calculate global average velocities
            if (rank == first)
            {
                double* average_velocity_foreign(new double[parameters.maxIters]);
                for (int i(last); i; --i)
                {
                    comm.Recv(average_velocity_foreign, parameters.maxIters, MPI::DOUBLE, MPI::ANY_SOURCE, 2);
                    for (int ii(0); ii < parameters.maxIters; ++ii)
                        average_velocity[ii] += average_velocity_foreign[ii];
                }
                delete[] average_velocity_foreign;
                for (int i(0); i < parameters.maxIters; ++i)
                    average_velocity[i] /= double(tot_cells);
            }
            else
                comm.Send(average_velocity, parameters.maxIters, MPI::DOUBLE, first, 2);

            comm.Barrier();
            if (rank == first)
            {
            #ifdef __GNUC__
                gettimeofday(&timstr, NULL);
                toc = timstr.tv_sec + timstr.tv_usec / 1000000.0;

                getrusage(RUSAGE_SELF, &ru);
                timstr = ru.ru_utime;
                usrtim = timstr.tv_sec + timstr.tv_usec / 1000000.0;
                timstr = ru.ru_stime;
                systim = timstr.tv_sec + timstr.tv_usec / 1000000.0;
            #else
                clock_end = std::chrono::high_resolution_clock::now();
            #endif
            }
        }
#ifdef TIMING
        delete[] average_velocity;
        if (rank == first)
        {
        #ifdef __GNUC__
            timing << size << " cores: " << toc - tic << " (s)\n";
            std::cout << size << " cores: " << toc - tic << " (s)\n" << std::flush;
        #else
            timing << size << " cores: " << std::chrono::duration<double>(clock_end - clock_start).count() << " (s)\n";
            std::cout << size << " cores: " << std::chrono::duration<double>(clock_end - clock_start).count() << " (s)\n" << std::flush;
        #endif
            timing.flush();
        }
    }
#else
    // Write final values //
    if (rank == first)
    {
        std::cout << "==done==\n" << std::flush
                  << "Reynolds number:\t\t" << average_velocity[parameters.maxIters - 1] * parameters.reynolds_dim / (1.0/3.0 / parameters.omega - 1.0/6.0) << '\n';
        std::cout.precision(6);
    #ifdef __GNUC__
        std::cout << "Elapsed time:\t\t\t" << toc - tic << " (s)\n"
                  << "Elapsed user CPU time:\t\t" << usrtim << " (s)\n"
                  << "Elapsed system CPU time:\t" << systim << " (s)\n";
    #else
        std::cout << "Elapsed time:\t\t\t" << std::chrono::duration<double>(clock_end - clock_start).count() << " (s)\n";
    #endif
    }

    write_values(comm, parameters, cells, obstacles, average_velocity, rank, size, first);

    // delete[] message_buffer;
    delete[] average_velocity;
#endif
    
    MPI::Finalize();
}
