/*
Code to implement a d2q9-bgk lattice Boltzmann scheme.
'd2' indicates a 2-dimensional grid, and
'q9' indicates 9 velocities per grid cell.
'bgk' refers to the Bhatnagar-Gross-Krook collision step.

The 'speeds' in each cell are numbered as follows:

4 7 1
 \|/
3-6-0
 /|\
5 8 2

A 2D grid:

          cols
      --- --- ---
     | D | E | F |
rows  --- --- ---
     | A | B | C |
      --- --- ---

'unwrapped' in row major order to give a 1D array:

 --- --- --- --- --- ---
| A | B | C | D | E | F |
 --- --- --- --- --- ---

Grid indices are:

        ny
        ^       cols(y)
        |  ----- ----- -----
        | | ... | ... | etc |
        |  ----- ----- -----
rows(y) | | 1,0 | 1,1 | 1,2 |
        |  ----- ----- -----
        | | 0,0 | 0,1 | 0,2 |
        |  ----- ----- -----
        ----------------------> nx

Note the names of the input parameter and obstacle files
are passed on the command line, e.g.:

  d2q9-bgk.exe input.params obstacles.dat

Be sure to adjust the grid dimensions in the parameter file
if you choose a different obstacle file.
*/

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <cmath>
#ifdef _OPENMP
    #include <omp.h>
#endif
#ifdef __GNUC__
    #include <sys/time.h>
    #include <sys/resource.h>
#else
    #include <chrono>
#endif

//#define tot_cells_once

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
        union
        {
            uint32_t u[2];
            double d;
        } t;

        t.u[0] = columns_in;
        t.u[1] = 0x43300000;
        t.d -= 4503599627370496.0;
        columns_log = (t.u[1] >> 20) - 0x3FF;

        data = new(std::nothrow) T[rows_in << columns_log];
    }

    ~Matrix()
    {
        delete[] data;
    }

    T* operator[](const int row) const
    {
        return &data[row << columns_log];
    }

    bool operator!()
    {
        return data == NULL;
    }

    T* data;
    int rows, columns;
    unsigned columns_log;
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

struct t_speed
{
    double speeds[NSPEEDS];
};

void write_values   (const t_param& params, const Matrix<t_speed>& cells, const Matrix<int>& obstacles, const double* av_vels);
double calc_reynolds(const t_param& params, const Matrix<t_speed>& cells, const Matrix<int>& obstacles);
double total_density(const t_param& params, const Matrix<t_speed>& cells);


//____________________________________________________________________________________________________//


int main(int argc, char* argv[])
{
    // Initialisation //
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <paramfile> <obstaclefile>\n";
        return EXIT_FAILURE;
    }

    char*& parameters_filepath = argv[1];
    char*& obstacles_filepath = argv[2];

    std::cout.sync_with_stdio(false);
    std::cerr.sync_with_stdio(false);
    std::cout.setf(std::ios_base::scientific, std::ios_base::floatfield);
    std::cout.setf(std::ios_base::uppercase);
    std::cout.precision(12);
#ifdef _OPENMP
    omp_set_dynamic(omp_get_max_threads());
#endif

    // Process parameter file
    t_param parameters;
    {
        std::ifstream parameters_file(parameters_filepath);

        if (!parameters_file)
        {
            std::cerr << "Could not open input parameter file " << parameters_filepath << '\n';
            return EXIT_FAILURE;
        }

        // Read in the parameter values
        parameters_file >> parameters.nx >> parameters.ny >> parameters.maxIters >> parameters.reynolds_dim >> parameters.density >> parameters.accel >> parameters.omega;
        if (!parameters_file)
        {
            std::cerr << "Invalid parameters in input parameter file\n";
            return EXIT_FAILURE;
        }
    }

    // Main grid
    Matrix<t_speed> cells_all[2] = {{parameters.ny, parameters.nx}, {parameters.ny, parameters.nx}};
    Matrix<t_speed>& cells = cells_all[0];
    if (!cells)
    {
        std::cerr << "Cannot allocate memory for cells\n";
        return EXIT_FAILURE;
    }

    double density_centre   = parameters.density * 4.0 / 9.0,
           density_axis     = parameters.density / 9.0,
           density_diagonal = parameters.density / 36.0;

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
        return EXIT_FAILURE;
    }

    // The map of obstacles
    Matrix<int> obstacles(parameters.ny, parameters.nx);
    if (!obstacles)
    {
        std::cerr << "Cannot allocate memory for obstacles\n";
        return EXIT_FAILURE;
    }

    // First set all cells in obstacle array to zero
    std::memset(obstacles.data, 0, obstacles.rows * obstacles.columns);

#ifdef tot_cells_once
    int tot_cells(obstacles.rows * obstacles.columns);
#endif
    // Process obstacle data file
    {
        std::ifstream obstacle_file(obstacles_filepath);
        if (!obstacle_file)
        {
            std::cerr << "Could not open input obstacle file " << obstacles_filepath << '\n';
            return EXIT_FAILURE;
        }

        int x, y, blocked;
        while (true)
        {
            obstacle_file >> x >> y >> blocked;
            if (!obstacle_file)
            {
                if (obstacle_file.eof())
                    break;
                std::cerr << "Invalid obstacle in input obstacle file\n";
                return EXIT_FAILURE;
            }

            if (x >= obstacles.columns)
            {
                std::cerr << "Invalid obstacle x co-ordinate in input obstacle file\n";
                return EXIT_FAILURE;
            }

            if (y >= obstacles.rows)
            {
                std::cerr << "Invalid obstacle y co-ordinate in input obstacle file\n";
                return EXIT_FAILURE;
            }

            if (blocked != 1)
            {
                std::cerr << "Invalid obstacle blocked value in input obstacle file. Required to be 1 - why am I even reading this value?!\n";
                return EXIT_FAILURE;
            }

            obstacles[y][x] = 1;
        #ifdef tot_cells_once
            --tot_cells;
        #endif
        }
    }

    // Allocate space to hold a record of the average velocities computed at each time step
    double* average_velocity(new double[parameters.maxIters]);

    double tot_u(0.0);
#ifndef tot_cells_once
    int tot_cells(0);
#endif
    int cells_switch(0);

#ifdef __GNUC__
    struct timeval timstr;
    double tic, toc, usrtim, systim;
    struct rusage ru;
#else
    std::chrono::system_clock::time_point clock_start, clock_end;
#endif

    // Main loop //
#pragma omp parallel
    {
    #pragma omp single
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
        #ifndef tot_cells_once
            tot_cells = 0;
        #endif
            
            // Accelerate flow //
        #pragma omp single
            {
                // Weights
                const double w1(parameters.density * parameters.accel / 9.0), // Speed centre row
                             w2(w1 / 4.0);                                    // Speed top/bottom rows

                // Modify the 2nd-from-top row of the grid
                const int y(cells.rows - 2);

                for (int x = 0; x < cells.columns; ++x)
                    if (!obstacles[y][x])
                    {
                        t_speed& cell(cells[y][x]);
                        if
                        (
                               cell.speeds[CL] - w1 > 0.0 // Centre left
                            && cell.speeds[TL] - w2 > 0.0 // Top left
                            && cell.speeds[BL] - w2 > 0.0 // Bottom left
                        )
                        {
                            cell.speeds[CL] -= w1; // Centre left
                            cell.speeds[TL] -= w2; // Top left
                            cell.speeds[BL] -= w2; // Bottom left
                            cell.speeds[CR] += w1; // Centre right
                            cell.speeds[TR] += w2; // Top right
                            cell.speeds[BR] += w2; // Bottom right
                        }
                    }
            }
        #pragma omp barrier

    #ifdef tot_cells_once
        #pragma omp for reduction(+ : tot_u)
    #else
        #pragma omp for reduction(+ : tot_u, tot_cells)
    #endif
            for (int y = 0; y < cells.rows; ++y) for (int x = 0; x < cells.columns; ++x)
            {
                // Propagate //
                {
                    const int y_n = y != cells.rows - 1    ? y + 1 : 0;
                    const int x_e = x != cells.columns - 1 ? x + 1 : 0;
                    const int y_s = y > 0                  ? y - 1 : cells.rows - 1;
                    const int x_w = x > 0                  ? x - 1 : cells.columns - 1;

                    // Propagate densities to neighbouring cells, following appropriate directions of travel and writing into scratch space grid
                    t_speed& tmp_cell(cells_temp[y][x]);
                    tmp_cell.speeds[CR] = cells[y]  [x_w].speeds[CR]; // Centre right
                    tmp_cell.speeds[TR] = cells[y_s][x_w].speeds[TR]; // Top    right
                    tmp_cell.speeds[BR] = cells[y_n][x_w].speeds[BR]; // Bottom right
                    tmp_cell.speeds[CL] = cells[y]  [x_e].speeds[CL]; // Centre left
                    tmp_cell.speeds[TL] = cells[y_s][x_e].speeds[TL]; // Top    left
                    tmp_cell.speeds[BL] = cells[y_n][x_e].speeds[BL]; // Bottom left
                    tmp_cell.speeds[CC] = cells[y]  [x]  .speeds[CC]; // Centre
                    tmp_cell.speeds[TC] = cells[y_s][x]  .speeds[TC]; // Top    centre
                    tmp_cell.speeds[BC] = cells[y_n][x]  .speeds[BC]; // Bottom centre
                }
            //}


        //#pragma omp for reduction(+ : tot_u, tot_cells)
            //for (int ii = 0; ii < cells.rows * cells.columns; ++ii)
            //{
                // Collision //
                {
                    // Rebound //
                    if (obstacles[y][x])
                    {
                        double* const& speeds = cells_temp[y][x].speeds;
                        std::swap(speeds[CR], speeds[CL]); // Centre right  <-> Centre left
                        std::swap(speeds[TC], speeds[BC]); // Top    centre <-> Bottom centre
                        std::swap(speeds[TR], speeds[BL]); // Top    right  <-> Bottom left
                        std::swap(speeds[TL], speeds[BR]); // Top    left   <-> Bottom right
                        continue;
                    }

                    t_speed& cell(cells[y][x]);
                    t_speed& tmp_cell(cells_temp[y][x]);

                    // compute local density total
                    double local_density(0.0);

                    for (int iii(0); iii < NSPEEDS; ++iii)
                        local_density += tmp_cell.speeds[iii];

                    // compute x velocity component
                    double u_x =
                        (
                              tmp_cell.speeds[CR]
                            + tmp_cell.speeds[TR]
                            + tmp_cell.speeds[BR]
                            - tmp_cell.speeds[CL]
                            - tmp_cell.speeds[TL]
                            - tmp_cell.speeds[BL]
                        )
                        / local_density;

                    // compute y velocity component
                    double u_y =
                        (
                              tmp_cell.speeds[TC]
                            + tmp_cell.speeds[TR]
                            + tmp_cell.speeds[TL]
                            - tmp_cell.speeds[BC]
                            - tmp_cell.speeds[BL]
                            - tmp_cell.speeds[BR]
                        )
                        / local_density;

                    // Equilibrium densities
                    double d_equ[NSPEEDS];
                    
                    {
                        double x, y, z, w;
                        const double w0(4.0 / 9.0 / 2.0),
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
                        tmp_cell.speeds[iii] = tmp_cell.speeds[iii] + parameters.omega*(d_equ[iii] * local_density - tmp_cell.speeds[iii]);


                    // Average velocity //
                    local_density = 0.0;

                    for (int iii(0); iii < NSPEEDS; ++iii)
                        local_density += tmp_cell.speeds[iii];

                    u_x =
                        (
                              tmp_cell.speeds[CR]
                            + tmp_cell.speeds[TR]
                            + tmp_cell.speeds[BR]
                            - tmp_cell.speeds[CL]
                            - tmp_cell.speeds[TL]
                            - tmp_cell.speeds[BL]
                        );

                    u_y =
                        (
                              tmp_cell.speeds[TC]
                            + tmp_cell.speeds[TR]
                            + tmp_cell.speeds[TL]
                            - tmp_cell.speeds[BC]
                            - tmp_cell.speeds[BL]
                            - tmp_cell.speeds[BR]
                        );

                    // Accumulate the norm of velocity
                    tot_u += std::sqrt(u_x*u_x + u_y*u_y) / local_density;
                #ifndef tot_cells_once
                    ++tot_cells;
                #endif
                }
            }

        #pragma omp single
            {
                average_velocity[i] = tot_u / (double)tot_cells;
            #ifdef DEBUG
                if (i % 1000 == 0)
                {
                    std::cout << "==timestep: " << i / 1000 << '/' << parameters.maxIters / 1000 << "==\n";
                    std::cout << "av velocity: " << average_velocity[i] << '\n';
                    std::cout << "tot density: " << total_density(parameters, cells) << '\n';
                }
            #endif
                cells_switch = 1 - cells_switch;
            }
        }

    #pragma omp single
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

    // Write final values //
    std::cout << "==done==\n"
              << "Reynolds number:\t\t" << calc_reynolds(parameters, cells, obstacles) << '\n';
    std::cout.precision(6);
#ifdef __GNUC__
    std::cout << "Elapsed time:\t\t\t" << toc - tic << " (s)\n"
              << "Elapsed user CPU time:\t\t" << usrtim << " (s)\n"
              << "Elapsed system CPU time:\t" << systim << " (s)\n";
#else
    std::cout << "Elapsed time:\t\t\t" << std::chrono::duration<double>(clock_end - clock_start).count() << " (s)\n";
#endif

    write_values(parameters, cells, obstacles, average_velocity);

    delete[] average_velocity;
}

double calc_reynolds(const t_param& params, const Matrix<t_speed>& cells, const Matrix<int>& obstacles)
{
    int tot_cells = 0;  // No. of cells used in calculation
    double tot_u = 0.0; // Accumulated magnitudes of velocity for each cell

    // loop over all non-blocked cells
    for (int y(0); y < cells.rows; ++y)
        for (int x(0); x < cells.columns; ++x)
        {
            // ignore occupied cells
            if (obstacles[y][x])
                continue;

            // local density total
            double local_density = 0.0;

            const t_speed& cell(cells[y][x]);

            for (int i(0); i < NSPEEDS; ++i)
                local_density += cell.speeds[i];

            // X-component of velocity
            double u_x =
                (
                      cell.speeds[CR]
                    + cell.speeds[TR]
                    + cell.speeds[BR]
                    - cell.speeds[CL]
                    - cell.speeds[TL]
                    - cell.speeds[BL]
                );

            // Y-component of velocity
            double u_y =
                (
                      cell.speeds[TC]
                    + cell.speeds[TR]
                    + cell.speeds[TL]
                    - cell.speeds[BC]
                    - cell.speeds[BL]
                    - cell.speeds[BR]
                );

            // Accumulate the norm of velocity
            tot_u += std::sqrt(u_x*u_x + u_y*u_y) / local_density;

            // Increase counter of inspected cells
            ++tot_cells;
        }

    const double viscosity = params.reynolds_dim / (1.0/3.0 / params.omega - 1.0/6.0);
    // const double viscosity = (12.0 / (2.0 - params.omega) - 6.0) * params.reynolds_dim; // Try this as well
    // const double viscosity = params.reynolds_dim / (1.0 / 6.0 * (2.0 / params.omega - 1.0)); // Original expression

    return viscosity * (tot_u / (double)tot_cells);
}

double total_density(const t_param& params, const Matrix<t_speed>& cells)
{
    double total = 0.0;

    for (int y(0); y < cells.rows; ++y)
        for (int x(0); x < cells.columns; ++x)
            for (int t(0); t < NSPEEDS; ++t)
                total += cells[y][x].speeds[t];

    return total;
}

void write_values(const t_param& params, const Matrix<t_speed>& cells, const Matrix<int>& obstacles, const double* av_vels)
{
    {
        std::ofstream file_state_file(FINALSTATEFILE);
        if (!file_state_file)
        {
            std::cerr << "Could not open output final state file " << FINALSTATEFILE << '\n';
            std::exit(EXIT_FAILURE);
        }

        file_state_file.setf(std::ios_base::scientific, std::ios_base::floatfield);
        file_state_file.setf(std::ios_base::uppercase);
        file_state_file.precision(12);

        for (int y(0); y < cells.rows; ++y)
            for (int x(0); x < cells.columns; ++x)
            {
                const double c_sq = 1.0 / 3.0; // sq. of speed of sound
                double pressure,               // fluid pressure in grid cell
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
                        local_density += cells[y][x].speeds[i];

                    // Compute x velocity component
                    u_x =
                        (
                              cells[y][x].speeds[CR]
                            + cells[y][x].speeds[TR]
                            + cells[y][x].speeds[BR]
                            - cells[y][x].speeds[CL]
                            - cells[y][x].speeds[TL]
                            - cells[y][x].speeds[BL]
                        )
                        / local_density;

                    // Compute y velocity component
                    u_y =
                        (
                              cells[y][x].speeds[TC]
                            + cells[y][x].speeds[TR]
                            + cells[y][x].speeds[TL]
                            - cells[y][x].speeds[BC]
                            - cells[y][x].speeds[BL]
                            - cells[y][x].speeds[BR]
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
    }

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
