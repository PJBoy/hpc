#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable
//#pragma OPENCL EXTENSION cl_khr_fp64 : enable

enum
{
    CR_e,
    TR_e,
    BR_e,
    CL_e,
    TL_e,
    BL_e,
    CC_e,
    TC_e,
    BC_e,
    NSPEEDS
};

typedef float my_float;

kernel void PropagateReboundCollision
(
    global my_float* const restrict cells,
    global my_float* const restrict cells_temp,
    global const char* const restrict obstacles,
    global long* const restrict tot_u,
    const int rows,
    const int columns,
    const my_float omega,
    const my_float densaccel
    /*
    , global int* const restrict DebugX
    , global int* const restrict DebugY
    //*/
)
{
    const int
        area = rows*columns,
        x = get_global_id(0),
        y = get_global_id(1);

    global my_float
        *const CR = cells_temp + CR_e * area + y*columns + x,
        *const TR = cells_temp + TR_e * area + y*columns + x,
        *const BR = cells_temp + BR_e * area + y*columns + x,
        *const CL = cells_temp + CL_e * area + y*columns + x,
        *const TL = cells_temp + TL_e * area + y*columns + x,
        *const BL = cells_temp + BL_e * area + y*columns + x,
        *const CC = cells_temp + CC_e * area + y*columns + x,
        *const TC = cells_temp + TC_e * area + y*columns + x,
        *const BC = cells_temp + BC_e * area + y*columns + x;
            
    // Rebound //
    if (obstacles[y*columns+x])
    {
        const int y_n = y != rows - 1    ? y + 1 : 0;
        const int x_e = x != columns - 1 ? x + 1 : 0;
        const int y_s = y > 0            ? y - 1 : rows - 1;
        const int x_w = x > 0            ? x - 1 : columns - 1;

        *CL = cells[CR_e * area + y  *columns+x_w];
        *BL = cells[TR_e * area + y_s*columns+x_w];
        *TL = cells[BR_e * area + y_n*columns+x_w];
        *CR = cells[CL_e * area + y  *columns+x_e];
        *BR = cells[TL_e * area + y_s*columns+x_e];
        *TR = cells[BL_e * area + y_n*columns+x_e];
        *CC = cells[CC_e * area + y  *columns+x]  ;
        *BC = cells[TC_e * area + y_s*columns+x]  ;
        *TC = cells[BC_e * area + y_n*columns+x]  ;
        return;
    }
    
    my_float speeds[NSPEEDS];
    
    // Propagate //
    {
        const int y_n = y != rows - 1    ? y + 1 : 0;
        const int x_e = x != columns - 1 ? x + 1 : 0;
        const int y_s = y > 0            ? y - 1 : rows - 1;
        const int x_w = x > 0            ? x - 1 : columns - 1;

        speeds[CR_e] = cells[CR_e * area + y  *columns+x_w];
        speeds[TR_e] = cells[TR_e * area + y_s*columns+x_w];
        speeds[BR_e] = cells[BR_e * area + y_n*columns+x_w];
        speeds[CL_e] = cells[CL_e * area + y  *columns+x_e];
        speeds[TL_e] = cells[TL_e * area + y_s*columns+x_e];
        speeds[BL_e] = cells[BL_e * area + y_n*columns+x_e];
        speeds[CC_e] = cells[CC_e * area + y  *columns+x]  ;
        speeds[TC_e] = cells[TC_e * area + y_s*columns+x]  ;
        speeds[BC_e] = cells[BC_e * area + y_n*columns+x]  ;
    }

    // Collision //
    {
        // compute local density total
        const my_float
            right = speeds[CR_e] + speeds[TR_e] + speeds[BR_e],
            left  = speeds[CL_e] + speeds[TL_e] + speeds[BL_e],
            up    = speeds[TC_e] + speeds[TR_e] + speeds[TL_e],
            down  = speeds[BC_e] + speeds[BR_e] + speeds[BL_e],
            local_density = speeds[CC_e] + speeds[TC_e] + speeds[BC_e] + right + left,
            u_x = (right - left) / local_density,
            u_y = (up - down) / local_density,
            u = u_x*u_x + u_y*u_y;

        // Equilibrium densities
        {
            my_float x, y, z;
            const my_float
                w2 = local_density * (1.0 / 8.0 / 9.0),
                w1 = w2 * 4.0,
                b = 1.0 - u*3.0;

                                 *CC = speeds[CC_e] + omega*((1.0 + b) * w1*4.0 - speeds[CC_e]);
            x = u_x * 3.0 + 1.0; *CR = speeds[CR_e] + omega*((x*x + b) * w1     - speeds[CR_e]);
            y = u_y * 3.0 + 1.0; *TC = speeds[TC_e] + omega*((y*y + b) * w1     - speeds[TC_e]);
            z = 2.0 - x;         *CL = speeds[CL_e] + omega*((z*z + b) * w1     - speeds[CL_e]);
            z = 2.0 - y;         *BC = speeds[BC_e] + omega*((z*z + b) * w1     - speeds[BC_e]);
            z = x + y - 1.0;     *TR = speeds[TR_e] + omega*((z*z + b) * w2     - speeds[TR_e]);
            z = 2.0 - z;         *BL = speeds[BL_e] + omega*((z*z + b) * w2     - speeds[BL_e]);
            z = y - x + 1.0;     *TL = speeds[TL_e] + omega*((z*z + b) * w2     - speeds[TL_e]);
            z = 2.0 - z;         *BR = speeds[BR_e] + omega*((z*z + b) * w2     - speeds[BR_e]);
        }
    
        // Accumulate the norm of velocity
        // This is prone to errors in the carry from bit31 to bit32, but it's the best that can be done without atom_add (and the carry cannot be taken into account with reliability)
        const ulong new_tot_u = sqrt(u) / local_density * 70368744177664.0f; // 2**46, 1024.0f*1024.0f*1024.0f*1024.0f*64.0f;
        atomic_add((global int*)tot_u, *(int*)&new_tot_u);
        atomic_add((global uint*)tot_u+1, *((uint*)(&new_tot_u)+1));
    }
    
    
    // Accelerate flow //
    // Modify the 2nd-from-top row of the grid
    if (y != 1)
        return;
    
    {
        // Weights
        const my_float
            w1 = densaccel,       // Speed centre row
            w2 = densaccel / 4.0; // Speed top/bottom rows
        if
        (
               cells_temp[CL_e * area + columns + x] - w1 > 0 // Centre left
            && cells_temp[TL_e * area + columns + x] - w2 > 0 // Top left
            && cells_temp[BL_e * area + columns + x] - w2 > 0 // Bottom left
        )
        {
            cells_temp[CL_e * area + columns + x] -= w1; // Centre left
            cells_temp[TL_e * area + columns + x] -= w2; // Top left
            cells_temp[BL_e * area + columns + x] -= w2; // Bottom left
            cells_temp[CR_e * area + columns + x] += w1; // Centre right
            cells_temp[TR_e * area + columns + x] += w2; // Top right
            cells_temp[BR_e * area + columns + x] += w2; // Bottom right
        }
    }
}
