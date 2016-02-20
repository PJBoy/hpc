#define __CL_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS

#include "CL/cl.hpp"
#include <omp.h>
#include <iostream>
#include <iomanip>
#include <array>
#include <iterator>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cmath>
#include <cerrno>
#ifdef __GNUC__
    #include <sys/time.h>
    #include <sys/resource.h>
#else
    #include <chrono>
#endif

//#ifndef __GNUC__
//#define DEBUG
//#endif
//#define DEBUG_EXCEPTIONS
//#define TIMING
#define tot_cells_once


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
    const int rows, columns;
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

typedef float my_float;

//____________________________________________________________________________________________________//


#ifdef DEBUG
namespace Debug
{
    std::ofstream TextFile;
    std::ofstream BinaryFile;
    std::ofstream ImageFile;

    void TextOutput()
    {
        if (!TextFile.is_open())
        {
            TextFile.open("Debug.txt", std::ios_base::out | std::ios_base::trunc);
            TextFile.flags(std::ios::uppercase | std::ios::hex);
            TextFile.fill('0');
        }
    }
    void BinaryOutput(void* data, int bytes = 1)
    {
        if (!BinaryFile.is_open())
            BinaryFile.open("Debug.bin", std::ios_base::out | std::ios_base::trunc | std::ios_base::binary);
        BinaryFile.write((char*)(data), bytes);
    }
    void BinaryOutput(void* data, void* end)
    {
        BinaryOutput(data, uintptr_t(end) - uintptr_t(data));
    }
    template<typename T>
    void ImageOutput(T* data, int width, int height)
    {
        if (!ImageFile.is_open())
            ImageFile.open("Debug.bmp", std::ios_base::out | std::ios_base::trunc | std::ios_base::binary);

        uint8_t BMPHeader[] = {0x42, 0x4D, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x38, 0x00, 0x00, 0x00, 0x28, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x18, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
        *((uint32_t*)(&BMPHeader[2])) = width*height + sizeof(BMPHeader);
        *((uint32_t*)(&BMPHeader[0x12])) = width;
        *((uint32_t*)(&BMPHeader[0x16])) = height;
        *((uint32_t*)(&BMPHeader[0x22])) = width*height;
        ImageFile.write((char*)(BMPHeader), sizeof(BMPHeader));

        float scale(255 / *std::max_element(data, data + width*height));
        int Padding((4 - width * 3 % 4) % 4);

        for (int y(height-1); y >= 0; --y)
        {
            for (int x(0); x != width; ++x)
            {
                uint8_t value(uint8_t(data[y*width + x] * scale));
                uint8_t BGR24[] = {value, value, value};
                ImageFile.write((char*)(BGR24), 3);
            }
            for (int x(Padding); x; --x)
                ImageFile.put(0);
        }
        ImageFile.close();
    }
}
#endif


const char* TranslateOpenCLError(cl_int errorCode)
{
    switch (errorCode)
    {
        case CL_SUCCESS:                            return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND:                   return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE:               return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE:             return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES:                   return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY:                 return "CL_OUT_OF_HOST_MEMORY";
        case CL_PROFILING_INFO_NOT_AVAILABLE:       return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case CL_MEM_COPY_OVERLAP:                   return "CL_MEM_COPY_OVERLAP";
        case CL_IMAGE_FORMAT_MISMATCH:              return "CL_IMAGE_FORMAT_MISMATCH";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case CL_BUILD_PROGRAM_FAILURE:              return "CL_BUILD_PROGRAM_FAILURE";
        case CL_MAP_FAILURE:                        return "CL_MAP_FAILURE";
        case CL_MISALIGNED_SUB_BUFFER_OFFSET:       return "CL_MISALIGNED_SUB_BUFFER_OFFSET";                          //-13
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:    return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";   //-14
        case CL_COMPILE_PROGRAM_FAILURE:            return "CL_COMPILE_PROGRAM_FAILURE";                               //-15
        case CL_LINKER_NOT_AVAILABLE:               return "CL_LINKER_NOT_AVAILABLE";                                  //-16
        case CL_LINK_PROGRAM_FAILURE:               return "CL_LINK_PROGRAM_FAILURE";                                  //-17
        case CL_DEVICE_PARTITION_FAILED:            return "CL_DEVICE_PARTITION_FAILED";                               //-18
        case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:      return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";                         //-19
        case CL_INVALID_VALUE:                      return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE_TYPE:                return "CL_INVALID_DEVICE_TYPE";
        case CL_INVALID_PLATFORM:                   return "CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE:                     return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT:                    return "CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES:           return "CL_INVALID_QUEUE_PROPERTIES";
        case CL_INVALID_COMMAND_QUEUE:              return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_HOST_PTR:                   return "CL_INVALID_HOST_PTR";
        case CL_INVALID_MEM_OBJECT:                 return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case CL_INVALID_IMAGE_SIZE:                 return "CL_INVALID_IMAGE_SIZE";
        case CL_INVALID_SAMPLER:                    return "CL_INVALID_SAMPLER";
        case CL_INVALID_BINARY:                     return "CL_INVALID_BINARY";
        case CL_INVALID_BUILD_OPTIONS:              return "CL_INVALID_BUILD_OPTIONS";
        case CL_INVALID_PROGRAM:                    return "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE:         return "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME:                return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL_DEFINITION:          return "CL_INVALID_KERNEL_DEFINITION";
        case CL_INVALID_KERNEL:                     return "CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX:                  return "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE:                  return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE:                   return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS:                return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION:             return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE:            return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE:             return "CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_GLOBAL_OFFSET:              return "CL_INVALID_GLOBAL_OFFSET";
        case CL_INVALID_EVENT_WAIT_LIST:            return "CL_INVALID_EVENT_WAIT_LIST";
        case CL_INVALID_EVENT:                      return "CL_INVALID_EVENT";
        case CL_INVALID_OPERATION:                  return "CL_INVALID_OPERATION";
        case CL_INVALID_GL_OBJECT:                  return "CL_INVALID_GL_OBJECT";
        case CL_INVALID_BUFFER_SIZE:                return "CL_INVALID_BUFFER_SIZE";
        case CL_INVALID_MIP_LEVEL:                  return "CL_INVALID_MIP_LEVEL";
        case CL_INVALID_GLOBAL_WORK_SIZE:           return "CL_INVALID_GLOBAL_WORK_SIZE";                           //-63
        case CL_INVALID_PROPERTY:                   return "CL_INVALID_PROPERTY";                                   //-64
        case CL_INVALID_IMAGE_DESCRIPTOR:           return "CL_INVALID_IMAGE_DESCRIPTOR";                           //-65
        case CL_INVALID_COMPILER_OPTIONS:           return "CL_INVALID_COMPILER_OPTIONS";                           //-66
        case CL_INVALID_LINKER_OPTIONS:             return "CL_INVALID_LINKER_OPTIONS";                             //-67
        case CL_INVALID_DEVICE_PARTITION_COUNT:     return "CL_INVALID_DEVICE_PARTITION_COUNT";                     //-68
            //    case CL_INVALID_PIPE_SIZE:                  return "CL_INVALID_PIPE_SIZE";                                  //-69
            //    case CL_INVALID_DEVICE_QUEUE:               return "CL_INVALID_DEVICE_QUEUE";                               //-70    

        default:
            return "UNKNOWN ERROR CODE";
    }
}


std::string get_file_contents(const char* filepath)
{
    std::ifstream f(filepath, std::ios::in | std::ios::binary);
    if (!f)
        throw(errno);
    
    std::string program;
    f.seekg(0, std::ios::end);
    program.resize(unsigned(f.tellg()));
    f.seekg(0, std::ios::beg);
    f.read(&program[0], program.size());
    f.close();

    return program;
}


double calc_reynolds(const t_param& params, const my_float* cells, const Matrix<char>& obstacles)
{
    int tot_cells = 0;  // No. of cells used in calculation
    double tot_u = 0.0; // Accumulated magnitudes of velocity for each cell
    const int
        columns(params.nx),
        rows(params.ny),
        area(rows * columns);

    // loop over all non-blocked cells
    for (int y(0); y < rows; ++y)
        for (int x(0); x < columns; ++x)
        {
            // ignore occupied cells
            if (obstacles[y][x])
                continue;

            // local density total
            double local_density = 0.0;

            for (int i(0); i < NSPEEDS; ++i)
                local_density += cells[i*area + y*columns + x];

            // X-component of velocity
            double u_x =
                (
                      cells[CR*area + y*columns + x]
                    + cells[TR*area + y*columns + x]
                    + cells[BR*area + y*columns + x]
                    - cells[CL*area + y*columns + x]
                    - cells[TL*area + y*columns + x]
                    - cells[BL*area + y*columns + x]
                );

            // Y-component of velocity
            double u_y =
                (
                      cells[TC*area + y*columns + x]
                    + cells[TR*area + y*columns + x]
                    + cells[TL*area + y*columns + x]
                    - cells[BC*area + y*columns + x]
                    - cells[BL*area + y*columns + x]
                    - cells[BR*area + y*columns + x]
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


double total_density(const t_param& params, const my_float* cells)
{
    double total = 0.0;

    for (const my_float* cell(cells); cell < cells + params.ny*params.nx*NSPEEDS; ++cells)
        total += *cell;

    return total;
}


void write_values(const t_param& params, const my_float* cells, const Matrix<char>& obstacles, const double* av_vels)
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
        
        const int
            columns(params.nx),
            rows(params.ny),
            area(rows * columns);
            
        for (int y(rows - 1); y >= 0; --y)
            for (int x(0); x < columns; ++x)
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
                        local_density += cells[i*area + y*columns + x];

                    // Compute x velocity component
                    u_x =
                        (
                              cells[CR*area + y*columns + x]
                            + cells[TR*area + y*columns + x]
                            + cells[BR*area + y*columns + x]
                            - cells[CL*area + y*columns + x]
                            - cells[TL*area + y*columns + x]
                            - cells[BL*area + y*columns + x]
                        ) / local_density;

                    // Compute y velocity component
                    u_y =
                        (
                              cells[BC*area + y*columns + x]
                            + cells[BL*area + y*columns + x]
                            + cells[BR*area + y*columns + x]
                            - cells[TC*area + y*columns + x]
                            - cells[TR*area + y*columns + x]
                            - cells[TL*area + y*columns + x]
                        ) / local_density;

                    // Compute norm of velocity
                    u = sqrt(u_x*u_x + u_y*u_y);

                    // Compute pressure
                    pressure = local_density * c_sq;
                }

                // write to file
                file_state_file << x << ' ' << y << ' ' << u_x << ' ' << u_y << ' ' << u << ' ' << pressure << ' ' << int(obstacles[y][x]) << '\n';
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


int main(int argc, char* argv[])
{
#ifndef DEBUG_EXCEPTIONS
    try
    {
#endif
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

    cl::Context context(CL_DEVICE_TYPE_GPU);
    cl::CommandQueue queue(context);
    cl::Program program(context, get_file_contents("template.cl"));
    program.build("-cl-finite-math-only -cl-fast-relaxed-math -cl-unsafe-math-optimizations");
    auto PropagateReboundCollision(cl::make_kernel
    <
        cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
        const int, const int, const my_float, const my_float
    >(program, "PropagateReboundCollision"));

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

    const int
        columns(parameters.nx),
        rows(parameters.ny),
        area(rows * columns);
        
    // Main grid
    my_float* cells_all[2] = {new my_float[area * NSPEEDS], new my_float[area * NSPEEDS]};
    my_float* cells = cells_all[0];
    if (!cells)
    {
        std::cerr << "Cannot allocate memory for cells\n";
        return EXIT_FAILURE;
    }

    const my_float
        density_centre   = my_float(parameters.density) * 4.0 / 9.0,
        density_axis     = my_float(parameters.density) / 9.0,
        density_diagonal = my_float(parameters.density) / 36.0;
        
    for (int y = 0; y < rows; ++y) for (int x = 0; x < columns; ++x) cells[CR*area + y*columns + x] = density_axis;
    for (int y = 0; y < rows; ++y) for (int x = 0; x < columns; ++x) cells[TR*area + y*columns + x] = density_diagonal;
    for (int y = 0; y < rows; ++y) for (int x = 0; x < columns; ++x) cells[BR*area + y*columns + x] = density_diagonal;
    for (int y = 0; y < rows; ++y) for (int x = 0; x < columns; ++x) cells[CL*area + y*columns + x] = density_axis;
    for (int y = 0; y < rows; ++y) for (int x = 0; x < columns; ++x) cells[TL*area + y*columns + x] = density_diagonal;
    for (int y = 0; y < rows; ++y) for (int x = 0; x < columns; ++x) cells[BL*area + y*columns + x] = density_diagonal;
    for (int y = 0; y < rows; ++y) for (int x = 0; x < columns; ++x) cells[CC*area + y*columns + x] = density_centre;
    for (int y = 0; y < rows; ++y) for (int x = 0; x < columns; ++x) cells[TC*area + y*columns + x] = density_axis;
    for (int y = 0; y < rows; ++y) for (int x = 0; x < columns; ++x) cells[BC*area + y*columns + x] = density_axis;

    // 'Helper' grid, used as scratch space
    my_float* cells_temp = cells_all[1];
    if (!cells_temp)
    {
        std::cerr << "Cannot allocate memory for temporary cells\n";
        return EXIT_FAILURE;
    }

    // The map of obstacles
    Matrix<char> obstacles(parameters.ny, parameters.nx);
    if (!obstacles)
    {
        std::cerr << "Cannot allocate memory for obstacles\n";
        return EXIT_FAILURE;
    }

    // First set all cells in obstacle array to zero
    std::memset(obstacles.data, 0, obstacles.rows * obstacles.columns * sizeof(*obstacles.data));

    int tot_cells(obstacles.rows * obstacles.columns);
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

            obstacles[obstacles.rows-1 - y][x] = 1;
            --tot_cells;
        }
    }

    cl::Buffer obstacles_buffer(context, obstacles[0], obstacles[obstacles.rows], true, true);

    // Allocate space to hold a record of the average velocities computed at each time step
    double* average_velocity(new double[parameters.maxIters]);

    /*
    Matrix<int> DebugX(parameters.ny, parameters.nx), DebugY(parameters.ny, parameters.nx);
    cl::Buffer
        DebugX_buffer(context, (int*)(DebugX[0]), (int*)(DebugX[DebugX.rows]), false, true),
        DebugY_buffer(context, (int*)(DebugY[0]), (int*)(DebugY[DebugY.rows]), false, true);
    //*/

    unsigned long tot_u(0);
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

        // Accelerate flow //
        // Modify the 2nd-from-top row of the grid
        for (int x(0); x < columns; ++x)
            if (!obstacles[1][x])
            {
                // Weights
                const my_float
                    w1 = parameters.density*parameters.accel / 9.0, // Speed centre row
                    w2 = w1 / 4.0;                                  // Speed top/bottom rows
                if
                (
                       cells[CL * area + columns + x] - w1 > 0 // Centre left
                    && cells[TL * area + columns + x] - w2 > 0 // Top left
                    && cells[BL * area + columns + x] - w2 > 0 // Bottom left
                )
                {
                    cells[CL * area + columns + x] -= w1; // Centre left
                    cells[TL * area + columns + x] -= w2; // Top left
                    cells[BL * area + columns + x] -= w2; // Bottom left
                    cells[CR * area + columns + x] += w1; // Centre right
                    cells[TR * area + columns + x] += w2; // Top right
                    cells[BR * area + columns + x] += w2; // Bottom right
                }
            }
            
        cl::Buffer cells_buffer_all[2] =
        {
            {context, cells     , cells      + area*NSPEEDS, false, true},
            {context, cells_temp, cells_temp + area*NSPEEDS, false, true}
        };
        

        for (int i(0); i < parameters.maxIters; ++i)
        {
            tot_u = 0;
            
        #pragma omp single
            {
                cl::Buffer tot_u_buffer(context, &tot_u, &tot_u + 1, false, true);
                
                PropagateReboundCollision
                (
                    cl::EnqueueArgs(queue, cl::NDRange(columns, rows), cl::NDRange(columns, 1)),
                    cells_buffer_all[cells_switch],
                    cells_buffer_all[1 - cells_switch],
                    obstacles_buffer,
                    tot_u_buffer,
                    rows,
                    columns,
                    my_float(parameters.omega),
                    my_float(parameters.density * parameters.accel / 9.0)
                    /*
                    , DebugX_buffer
                    , DebugY_buffer
                    //*/
                );
                cl::copy(queue, tot_u_buffer, &tot_u, &tot_u + 1);
                // cl::copy(queue, DebugX_buffer, (int*)(DebugX[0]), (int*)(DebugX[DebugX.rows]));
                // cl::copy(queue, DebugY_buffer, (int*)(DebugY[0]), (int*)(DebugY[DebugY.rows]));
            }

    #pragma omp single
        {
            average_velocity[i] = tot_u * parameters.density / (tot_cells * 1024.0*1024.0*1024.0*1024.0*64.0);
        #ifdef DEBUG
            if (i % 1000 == 0)
            {
                /*
                if (i == 0)
                {
                    Debug::ImageOutput(DebugY.data, DebugY.columns, DebugY.rows);
                    Debug::TextOutput();
                    for (int y(0); y < DebugY.rows; ++y)
                    {
                        for (int x(0); x < DebugY.columns; ++x)
                            Debug::TextFile << '(' << std::setw(2) << DebugX[y][x] << ',' << std::setw(2) << DebugY[y][x] << "); ";
                        Debug::TextFile << '\n';
                    }
                }
                //*/
                std::cout << "==timestep: " << i / 1000 << '/' << parameters.maxIters / 1000 << "==\n";
                std::cout << "av velocity: " << average_velocity[i] << std::endl;
            }
        #endif
            cells_switch = 1 - cells_switch;
        }
    }

    cl::copy(queue, cells_buffer_all[cells_switch], cells, cells + area*NSPEEDS);
    
    // Decelerate flow //
    // Modify the 2nd-from-top row of the grid
    for (int x(0); x < columns; ++x)
        if (!obstacles[1][x])
        {
            // Weights
            const my_float
                w1 = parameters.density*parameters.accel / 9.0, // Speed centre row
                w2 = w1 / 4.0;                                  // Speed top/bottom rows
            if
            (
                   cells[CL * area + columns + x] > 0 // Centre left
                && cells[TL * area + columns + x] > 0 // Top left
                && cells[BL * area + columns + x] > 0 // Bottom left
            )
            {
                cells[CL * area + columns + x] += w1; // Centre left
                cells[TL * area + columns + x] += w2; // Top left
                cells[BL * area + columns + x] += w2; // Bottom left
                cells[CR * area + columns + x] -= w1; // Centre right
                cells[TR * area + columns + x] -= w2; // Top right
                cells[BR * area + columns + x] -= w2; // Bottom right
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
    delete[] cells_all[1];
    delete[] cells_all[0];
#ifndef DEBUG_EXCEPTIONS
    }
    catch (cl::Error e)
    {
        std::cerr << TranslateOpenCLError(e.err()) << " in " << e.what() << std::endl;
    }
#endif
}
