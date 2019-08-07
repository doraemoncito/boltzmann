#ifdef CL_KHR_FP32
#define double	float
#else
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

#define NSPEEDS 9

#define CELL(ptr,size,idx,speed)    ptr[(speed)*(size)+(idx)]

/**
 * Accelerates the flow by modifying the first column of the grid.
 */
__kernel void accelerate_flow_kernel(__global double *cells, __global int *obstacles) {
	double normal[NSPEEDS];

	for (size_t gid = get_global_id(0) * params_nx; gid < size; gid += size) {
		for (unsigned int kk = 0; kk < NSPEEDS; kk++) {
			normal[kk] = CELL(cells, size, gid, kk);
		}

		/* increase 'east-side' densities */
		normal[1] += wa1;
		normal[5] += wa2;
		normal[8] += wa2;

		/* decrease 'west-side' densities */
		normal[3] -= wa1;
		normal[6] -= wa2;
		normal[7] -= wa2;

		/* if the cell is not occupied and we don't send a density negative */
		if ((!obstacles[gid]) && (normal[3] > 0.0) && (normal[6] > 0.0) && (normal[7] > 0.0)) {
			for (unsigned int kk = 0; kk < NSPEEDS; kk++) {
				CELL(cells, size, gid, kk) = normal[kk];
			}
		}
	}
}

/* weighting factor 4.0 / 9.0 */
#define w0   0.4444444444444444

/* weighting factor 1.0 / 9.0 */
#define w1   0.1111111111111111

/* weighting factor 1.0 / 36.0 */
#define w2   0.0277777777777778

/*
 * A combined and rebound and collision function.
 */
__kernel void collision_kernel(__global double *src_cells, __global double *dst_cells, __global int *obstacles,
	__global double *av_vels, __local double *scratch, unsigned int iteration_offset)
{
	__private double u[NSPEEDS];              /* directional velocities */
	__private double d_equ[NSPEEDS];          /* equilibrium densities */
	__private double normal[NSPEEDS];
	__private double mirror[NSPEEDS];
	double u_x, u_y;                           /* av. velocities in x and y directions */
	double u_sq;                              /* squared velocity */
	double local_density;                     /* sum of densities in a particular cell */
	unsigned int ii, jj;					  /* generic counters */
	unsigned int x_e, x_w, y_n, y_s;		  /* indices of neighbouring cells */

	const int tid = get_local_id(0);
	const int gsize = min((int) size, (int)((get_group_id(0) + 1) * get_local_size(0) * numStrides + tid));
	scratch[tid] = 0.0;

	for (int gid = get_group_id(0) * get_local_size(0) * numStrides + tid; gid < gsize; gid += get_local_size(0)) {

			/* -- COLLISION STEP --- */

			ii = gid / params_nx;
			jj = gid % params_nx;

			/* determine indices of axis-direction neighbours respecting periodic boundary conditions (wrap around) */
			y_n = (ii == 0) ? (params_ny - 1) : (ii - 1);
			x_e = (jj == 0) ? (params_nx - 1) : (jj - 1);
			y_s = (ii + 1) % params_ny;
			x_w = (jj + 1) % params_nx;

			/* propagate densities to neighbouring cells, following appropriate directions of travel and writing into scratch space grid */
			normal[0] = CELL(src_cells, size,  ii * params_nx + jj , 0); /* central cell, no movement */
			normal[1] = CELL(src_cells, size,  ii * params_nx + x_e, 1); /* east */
			normal[2] = CELL(src_cells, size, y_n * params_nx + jj , 2); /* north */
			normal[3] = CELL(src_cells, size,  ii * params_nx + x_w, 3); /* west */
			normal[4] = CELL(src_cells, size, y_s * params_nx + jj , 4); /* south */
			normal[5] = CELL(src_cells, size, y_n * params_nx + x_e, 5); /* north-east */
			normal[6] = CELL(src_cells, size, y_n * params_nx + x_w, 6); /* north-west */
			normal[7] = CELL(src_cells, size, y_s * params_nx + x_w, 7); /* south-west */
			normal[8] = CELL(src_cells, size, y_s * params_nx + x_e, 8); /* south-east */

			/* for cells that contain an obstacle */
			mirror[0] = normal[0];
			mirror[1] = normal[3];
			mirror[2] = normal[4];
			mirror[3] = normal[1];
			mirror[4] = normal[2];
			mirror[5] = normal[7];
			mirror[6] = normal[8];
			mirror[7] = normal[5];
			mirror[8] = normal[6];

			/* for unoccupied cells */
			/* compute local density total */
			local_density = normal[0];
			for (int kk = 1; kk < NSPEEDS; kk++) {
				local_density += normal[kk];
			}

			/* compute x velocity component */
			u_x = (normal[1] + normal[5] + normal[8] - (normal[3] + normal[6] + normal[7])) / local_density;

			/* compute y velocity component */
			u_y = (normal[2] + normal[5] + normal[6] - (normal[4] + normal[7] + normal[8])) / local_density;

			/* velocity squared */
			u_sq = u_x * u_x + u_y * u_y;

			/* directional velocity components */
			u[0] =   0.0;        /* central */
			u[1] =   u_x;        /* east */
			u[2] =         u_y;  /* north */
			u[3] = - u_x;        /* west */
			u[4] =       - u_y;  /* south */
			u[5] =   u_x + u_y;  /* north-east */
			u[6] = - u_x + u_y;  /* north-west */
			u[7] = - u_x - u_y;  /* south-west */
			u[8] =   u_x - u_y;  /* south-east */

			/* equilibrium densities */
			/* zero velocity density: weight w0 */
			d_equ[0] = w0;
			/* axis speeds: weight w1 */
			d_equ[1] = w1;
			d_equ[2] = w1;
			d_equ[3] = w1;
			d_equ[4] = w1;
			/* diagonal speeds: weight w2 */
			d_equ[5] = w2;
			d_equ[6] = w2;
			d_equ[7] = w2;
			d_equ[8] = w2;

			/* relaxation step */
			for (int kk = 0; kk < NSPEEDS; kk++) {
				d_equ[kk] = d_equ[kk] * local_density * (1.0 - u_sq * 1.5 + u[kk] * 3.0 + u[kk] * u[kk] * 4.5);
				normal[kk] = (!obstacles[gid]) ? (normal[kk] + params_omega * (d_equ[kk] - normal[kk])) : mirror[kk];
				CELL(dst_cells, size, gid, kk) = normal[kk];
			}

			/* -- REDUCTION STEP --- */

			/* compute local density total */
			local_density = normal[0];
			for (int kk = 1; kk < NSPEEDS; kk++) {
				local_density += normal[kk];
			}

			if (!obstacles[gid]) { scratch[tid] += (normal[1] + normal[5] + normal[8] - (normal[3] + normal[6] + normal[7])) / local_density; }

			/* -- ACCELERATION STEP -- */
			/* decrease 'west-side' densities */
			normal[3] -= wa1;
			normal[6] -= wa2;
			normal[7] -= wa2;

			/* If the cell in the first column is not occupied and we don't send a density negative... */
			if ((iteration_offset) && (0 == jj) && (!obstacles[gid]) && (normal[3] > 0.0) && (normal[6] > 0.0) && (normal[7] > 0.0)) {

				/* increase 'east-side' densities */
				normal[1] += wa1;
				normal[5] += wa2;
				normal[8] += wa2;

				for (unsigned int kk = 0; kk < NSPEEDS; kk++) {
					CELL(dst_cells, size, gid, kk) = normal[kk];
				}
			}
	}

	/* ...and finally perform the parallel reduction of all cells in the workgroup */
	barrier(CLK_LOCAL_MEM_FENCE);

	/* ...and finally perform the parallel reduction of all cells in the workgroup as described in the OpenCL reduction example in the
	 * Nvidia GPU computing SDK.  This essentially avoids unnecesary synchronisation on NVidia architectures.
	 */
#if 0
	for (int stride = get_local_size(0) >> 1; stride > 0; stride >>= 1) {
		if (tid < stride) scratch[tid] += scratch[tid + stride];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
#else
	const unsigned int blockSize = get_local_size(0);

    if (blockSize >=1024) { if (tid < 512) { scratch[tid] += scratch[tid + 512]; } barrier(CLK_LOCAL_MEM_FENCE); }
    if (blockSize >= 512) { if (tid < 256) { scratch[tid] += scratch[tid + 256]; } barrier(CLK_LOCAL_MEM_FENCE); }
    if (blockSize >= 256) { if (tid < 128) { scratch[tid] += scratch[tid + 128]; } barrier(CLK_LOCAL_MEM_FENCE); }
    if (blockSize >= 128) { if (tid <  64) { scratch[tid] += scratch[tid +  64]; } barrier(CLK_LOCAL_MEM_FENCE); }

    if (tid < 32) {
        if (blockSize >=  64) { scratch[tid] += scratch[tid + 32]; }
        if (blockSize >=  32) { scratch[tid] += scratch[tid + 16]; }
        if (blockSize >=  16) { scratch[tid] += scratch[tid +  8]; }
        if (blockSize >=   8) { scratch[tid] += scratch[tid +  4]; }
        if (blockSize >=   4) { scratch[tid] += scratch[tid +  2]; }
        if (blockSize >=   2) { scratch[tid] += scratch[tid +  1]; }
    }
#endif

	if (tid == 0) av_vels[iteration_offset + params_maxIters * get_group_id(0)] = scratch[0];
}


__kernel void final_reduction_kernel(__global double *av_vels, const unsigned int total_cells, const unsigned int workgroup_size) {
	int gid = get_global_id(0); // * workgroup_size;
	double average = av_vels[gid];
	for (int kk = 1; kk < workgroup_size; kk++) {
		average += av_vels[gid + params_maxIters * kk];
	}
	av_vels[gid] = average / total_cells;
}
