/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
**
** The 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** A 2D grid 'unwrapped' in row major order to give a 1D array:
**
**           cols
**       --- --- ---
**      | D | E | F |
** rows  --- --- ---
**      | A | B | C |
**       --- --- ---
**
**  --- --- --- --- --- ---
** | A | B | C | D | E | F |
**  --- --- --- --- --- ---
*/

#ifdef _WIN32
#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#else
#include <unistd.h>
#endif
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <OpenCL/opencl.h>

#ifdef CL_KHR_FP32
#define cl_double		cl_float
#define SCANF_FMT		"%f\n"
#else
#define SCANF_FMT		"%lf\n"
#endif /* CL_KHR_FP32 */

#ifdef DEBUG
#define TRACE(x)		debug x
#else
#define TRACE(x)
#endif /* DEBUG */

#define NSPEEDS         9
#define PARAMFILE       "input.params"
#define OBSTACLEFILE    "obstacles.dat"
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"

/* struct to hold the parameter values */
typedef struct {
	cl_uint   nx;            /* no. of cells in y-deirection */
	cl_uint   ny;            /* no. of cells in x-direction */
	cl_uint   maxIters;      /* no. of iterations */
	cl_uint   reynolds_dim;  /* dimension for Reynolds number */
	cl_uint   tot_cells;     /* total number of free (non-blocked) cells */
	cl_double density;       /* density per link */
	cl_double accel;         /* density redistribution */
	cl_double omega;         /* relaxation parameter */
} t_param;


#define CELL(ptr,size,idx,speed)    ptr[(speed)*(size)+(idx)]


extern double wtime();       // returns time since some fixed past point (wtime.c)
extern int output_platform_info(cl_platform_id platform_id);
extern int output_device_info(cl_device_id);
extern int err_code (cl_int err_in);
extern const char *err_code_str(cl_int err_in);

/*
** function prototypes
*/

/* load parameters */
void initialise_params(t_param* params);
/* allocate memory, load obstacles & initialize fluid particle densities */
void initialise_memory(t_param* params, cl_double** cells_ptr, cl_int** obstacles_ptr, cl_double** av_vels_ptr, cl_uint workgroups);

/* finalise, including freeing up allocated memory */
void finalise(const t_param* params, cl_double** cells_ptr, cl_int** obstacles_ptr, cl_double** av_vels_ptr);
void write_values(t_param params, cl_double* cells, cl_int* obstacles, cl_double* av_vels, cl_uint workgroups);

/* Sum all the densities in the grid.
 ** The total should remain constant from one timestep to the next. */
cl_double total_density(t_param params, cl_double* cells);

/* calculate Reynolds number */
cl_double calc_reynolds(t_param params, cl_double av_velocity);

/* utility functions */
void die(const char* message, int line, const char *file);

void debug(const char *format, ...);

#define NUM_PLATFORMS   4     // maximum number of different OpenCL platforms
#define NUM_DEVICES     4     // maximum number of different OpenCL devices

void setKernelArguments(cl_kernel kernel, ...) {
	int err = 0;
  	cl_uint arg_index = 0;
	va_list args;
	cl_mem argument;

	char actual_kernel_name[128] = "";
	cl_uint actual_num_kernel_arguments = 0;

	va_start(args, kernel);
	while ((argument = va_arg(args, cl_mem)) != NULL) {
	    err = clSetKernelArg(kernel, arg_index, sizeof(cl_mem), &argument);
		if (err != CL_SUCCESS) {
		    printf("Error: Failed to set kernel argument %d! %s\n", arg_index, err_code_str(err));
			fflush(stdout);
			exit(1);
		}
		arg_index++;
	}
	va_end(args);

	clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, sizeof(actual_kernel_name), actual_kernel_name, NULL);
	clGetKernelInfo(kernel, CL_KERNEL_NUM_ARGS, sizeof(cl_uint), &actual_num_kernel_arguments, NULL);
	TRACE(("Kernel %s had %d arguments set\n", actual_kernel_name, actual_num_kernel_arguments));
}

cl_kernel createKernel(cl_program program, const char *name) {
	int err = 0;
	cl_kernel kernel = clCreateKernel(program, name, &err);
    if (!kernel || err != CL_SUCCESS) {
        printf("Error: failed to create collision kernel! %s\n", err_code_str(err));
		fflush(stdout);
        exit(1);
    }
	return kernel;
}

cl_mem createBuffer(cl_context context, size_t size, void *host_ptr) {
	int err = 0;
	cl_mem result = clCreateBuffer(context, CL_MEM_READ_WRITE | ((host_ptr != NULL) ? CL_MEM_USE_HOST_PTR : 0), size, host_ptr, &err);
	if ((!result) || (err != CL_SUCCESS)) {
		printf("Error: Failed to allocate device memory! %s\n", err_code_str(err));
		fflush(stdout);
		exit(1);
	}
	return result;
}

void enqueueWriteBuffer(cl_command_queue command_queue, cl_mem buffer, size_t cb, const void *ptr) {
	cl_int err = clEnqueueWriteBuffer(command_queue, buffer, CL_FALSE, 0, cb, ptr, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		printf("Error: Failed to write memory buffer to device!\n");
		fflush(stdout);
		exit(1);
	}
}

char *readProgramFromFile(const char *filename) {
   char *program_buffer;
   size_t program_size;
   FILE *program_handle = fopen(filename, "r");
   if(program_handle == NULL) {
      perror("Couldn't find the program file");
      exit(1);
   }
   fseek(program_handle, 0, SEEK_END);
   program_size = ftell(program_handle);
   rewind(program_handle);
   program_buffer = (char*) calloc(1, program_size + 1);
   fread(program_buffer, 1, program_size, program_handle);
   fclose(program_handle);
   return program_buffer;
}

void getNumBlocksAndThreads(cl_uint n, cl_uint stride, cl_uint maxComputeUnits, size_t maxWorkGroupSize, size_t *global, size_t *local) {
	/* By default we execute the kernel over the entire range of our 1d input data set
     * using the maximum number of work group items for this device.
	 */
	*local = maxWorkGroupSize;
	*global = n / stride;
	*global = *global + maxWorkGroupSize - *global % maxWorkGroupSize;
}


char buildOptions[1024] = "";


int main(int argc, char** argv) {
    cl_int err;						/* error code returned from OpenCL calls */

	char *kernel_source = NULL;
	cl_uint maxComputeUnits = 0;
	size_t maxWorkGroupSize = 0;

    size_t global = 0;				/* global domain size */
    size_t local = 0;				/* local domain size */
    cl_uint workgroups = 0;

    cl_device_id device_id;			/* compute device Id */
    cl_context context;           // compute context
    cl_command_queue commands;    // compute command queue
    cl_program program;           // compute program

    cl_kernel accelerate_flow_kernel;
    cl_kernel collision_kernel;
    cl_kernel final_reduction_kernel;

    cl_mem cl_obstacles;
    cl_mem cl_cells;
	cl_mem cl_dst_cells;
	cl_mem cl_av_vels;

	cl_uint ii = 0;
	size_t iteration_offset = 0;

	// Discover the list of available OpenCL platforms and devices per-platform
    cl_uint num_platforms;
    cl_platform_id platforms[NUM_PLATFORMS];
    cl_uint num_devices[NUM_PLATFORMS];
    cl_device_id devices[NUM_PLATFORMS][NUM_DEVICES];

	// Connect to platform AMD (platform=0) or platform Intel (platform=1)
    const int platform = 0;
	// Connect to a CPU (gpu=0) or a GPU (gpu=1)
    const int gpu = 1;

	cl_uint numStrides = 9;
	double rtime;

    t_param params;					/* struct to hold parameter values */
    cl_double* cells = NULL;        /* source grid containing fluid densities */
    cl_int *obstacles = NULL;     /* grid indicating which cells are blocked */
	cl_double *av_vels  = NULL;		/* a record of the av. velocity computed for each timestep */

#ifdef DEBUG
    printf("sizeof(double):         %lu\n", sizeof(double));
    printf("sizeof(cl_double):      %lu\n", sizeof(cl_double));
    printf("sizeof(int):            %lu\n", sizeof(int));
    printf("sizeof(cl_int):         %lu\n", sizeof(cl_int));
    printf("sizeof(unsigned char):  %lu\n", sizeof(unsigned char));
    printf("sizeof(cl_uchar):       %lu\n", sizeof(cl_uchar));
    printf("sizeof(void *):         %lu\n", sizeof(void *));
    printf("sizeof(t_param):        %lu\n", sizeof(t_param));
#endif

	printf("d2q9-bgk lattice Boltzmann scheme: OpenCL\n");
	/* Initialize our data structures and load values from file */
    initialise_params(&params);
	printf("Performing %d iterations on a grid of size %d x %d\n", params.maxIters, params.ny, params.nx);

	// Just get the first available platform for this example
    err = clGetPlatformIDs (NUM_PLATFORMS, &platforms[platform], &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        printf("Error: Didn't find any OpenCL platforms!\n");
        return EXIT_FAILURE;
    }

    printf("[DEBUG] Found %d OpenCL platforms\n", num_platforms);
    for (unsigned int platformIndex = 0; platformIndex < num_platforms; platformIndex++) {
        printf("[DEBUG] Platform %d:\n", platformIndex);
        err = output_platform_info(platforms[platformIndex]);
        if (clGetDeviceIDs(platforms[platformIndex], (gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU),
                NUM_DEVICES, devices[platformIndex], &num_devices[platformIndex]) == CL_SUCCESS) {
            for (unsigned int deviceIndex = 0; deviceIndex < num_devices[platformIndex]; deviceIndex++) {
                printf("[DEBUG] Platform %d, Device %d\n", platformIndex, deviceIndex);
                err = output_device_info(devices[platformIndex][deviceIndex]);
            }
        }
    }

    printf("[INFO] SELECTED PLATFORM AND DEVICE:\n");
    err = output_platform_info(platforms[platform]);
    // Just get the first available device for this example
    device_id = devices[0][0];
    err = output_device_info(device_id);

    // Create a compute context
	TRACE(("Creating a compute context\n"));
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if ((!context) || (err != CL_SUCCESS)) {
        printf("Error: Failed to create a compute context! %s\n", err_code_str(err));
        return EXIT_FAILURE;
    }

	err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &maxComputeUnits, NULL);
	if (err != CL_SUCCESS) {
		printf("Error: Failed to retrieve number of compute units! %s\n", err_code_str(err));
		exit(1);
	}

	// Create a command queue
    commands = clCreateCommandQueue(context, device_id, 0, &err);
    if ((!commands) || (err != CL_SUCCESS)) {
        printf("Error: Failed to create a command commands! %s\n", err_code_str(err));
        return EXIT_FAILURE;
    }

    // Create the compute program from the source buffer
	kernel_source = readProgramFromFile("d2q9-bgk.cl");
#ifdef DEBUG
//	TRACE(("--- BEGIN ----\n%s\n--- END ---\n", kernel_source));
#endif
	program = clCreateProgramWithSource(context, 1, (const char **) &kernel_source, NULL, &err);
    if ((!program) || (err != CL_SUCCESS)) {
        printf("Error: Failed to create compute program! %s\n", err_code_str(err));
        return EXIT_FAILURE;
    }

    /* Pass in the grid size to the program as a compiler macro to speed up execution. */
    sprintf(buildOptions,
#ifdef CL_KHR_FP32
			"-DCL_KHR_FP32 "
#endif /* CL_KHR_FP32 */
    		"-cl-no-signed-zeros -cl-unsafe-math-optimizations -cl-mad-enable "
    		"-Dsize=%d -Dparams_nx=%d -Dparams_ny=%d -Dparams_maxIters=%d -Dparams_density=%lf "
    		"-Dparams_accel=%lf -Dparams_omega=%lf -DnumStrides=%d -Dwa1=%.24lf -Dwa2=%.24lf",
    		params.nx * params.ny, params.nx, params.ny, params.maxIters, params.density, params.accel, params.omega,
    		numStrides, params.density * params.accel / 9.0, params.density * params.accel / 36.0);

    printf("Build options: %s\n", buildOptions);
	err = clBuildProgram(program, 0, NULL, buildOptions, NULL, NULL);
	if (err != CL_SUCCESS) {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    }

    // Create the compute kernels from the program
    accelerate_flow_kernel = createKernel(program, "accelerate_flow_kernel");
    collision_kernel = createKernel(program, "collision_kernel");
    final_reduction_kernel = createKernel(program, "final_reduction_kernel");

	/* Get the maximum work group size for executing the kernel on the device */
    err = clGetKernelWorkGroupInfo(collision_kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkGroupSize, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        exit(1);
    }

	printf("grid size %d, stride %d, compute units %u, max workgroup size %ld\n",
			params.ny * params.nx, numStrides, maxComputeUnits, maxWorkGroupSize);
	/* WARNING: maximum WorkGroup size must be a power of two and multiple of the warp size for the reduction to work */
	getNumBlocksAndThreads(params.ny * params.nx, numStrides, maxComputeUnits, 512 /*maxWorkGroupSize*/, &global, &local);
	workgroups = global / local;
	printf("[NDRANGE] global: %6ld, local: %4ld, number of workgroups: %4d\n", global, local, workgroups);

	/* Initialize our data structures and load values from file */
    initialise_memory(&params, &cells, &obstacles, &av_vels, workgroups);

    cl_cells = createBuffer(context, sizeof(cl_double) * NSPEEDS * params.nx * params.ny, cells);
    cl_dst_cells = createBuffer(context, sizeof(cl_double) * NSPEEDS * params.nx * params.ny, NULL);
    cl_obstacles = createBuffer(context, sizeof(cl_int) * params.nx * params.ny, obstacles);
    cl_av_vels = createBuffer(context, sizeof(cl_double) * params.maxIters * workgroups, NULL);

	enqueueWriteBuffer(commands, cl_cells, sizeof(cl_double) * NSPEEDS * params.nx * params.ny, cells);
	enqueueWriteBuffer(commands, cl_obstacles, sizeof(cl_int) * params.nx * params.ny, obstacles);

	// Set the arguments to our compute kernels
	setKernelArguments(accelerate_flow_kernel, cl_cells, cl_obstacles, NULL);

	setKernelArguments(collision_kernel, cl_dst_cells, cl_cells, cl_obstacles, cl_av_vels, NULL);
	clSetKernelArg(collision_kernel, 4, sizeof(cl_double) * local, NULL);

	setKernelArguments(final_reduction_kernel, cl_av_vels, NULL);
	clSetKernelArg(final_reduction_kernel, 1, sizeof(cl_uint), &(params.tot_cells));
	clSetKernelArg(final_reduction_kernel, 2, sizeof(cl_uint), &workgroups);

#ifdef _WIN32
	timeBeginPeriod(1);
#endif
	rtime = wtime();

	/*
	 * Accelerate for the first time.  Subsequent acceleration steps will take place inside the collision step (at the very end)
	 */
	TRACE(("Queuing accelerate_flow_kernel, domain size { %d, %d }\n", global, local));
	err = clEnqueueNDRangeKernel(commands, accelerate_flow_kernel, 1, NULL, &local, &local, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
	  printf("Error: Failed to execute accelerate flow kernel! %s\n", err_code_str(err));
	  exit(1);
	}

	/* iterate for maxIters timesteps */
    for (ii = 0; ii < params.maxIters; ii++) {
		TRACE(("==timestep: %d==\n", ii));

		// Count down to zero so that test inside the kernel is more efficient
		iteration_offset = (params.maxIters - 1 - ii);
		TRACE(("Queuing collision_kernel, domain size { %d, %d }\n", global, local));
		err  = clSetKernelArg(collision_kernel, 0, sizeof(cl_mem), (((ii % 2) == 0) ? &cl_cells : &cl_dst_cells));
		err |= clSetKernelArg(collision_kernel, 1, sizeof(cl_mem), (((ii % 2) == 0) ? &cl_dst_cells : &cl_cells));
		err |= clSetKernelArg(collision_kernel, 5, sizeof(cl_uint), &iteration_offset);
		err |= clEnqueueNDRangeKernel(commands, collision_kernel, 1, NULL, &global, &local, 0, NULL, NULL);
		if (err != CL_SUCCESS) {
		  printf("Error: Failed to execute collision kernel! %s\n", err_code_str(err));
		  exit(1);
		}
	}

	global = params.maxIters;
	TRACE(("Performing final reduction, domain size { %d, NULL }\n", global));
	err = clEnqueueNDRangeKernel(commands, final_reduction_kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
	  printf("Error: Failed to execute final reduction kernel! %s\n", err_code_str(err));
	  exit(1);
	}

	// Read back the results from the compute device
	err  = clEnqueueReadBuffer(commands, cl_cells, CL_TRUE, 0, sizeof(cl_double) * NSPEEDS * params.nx * params.ny, cells, 0, NULL, NULL);
	err |= clEnqueueReadBuffer(commands, cl_av_vels, CL_TRUE, 0, sizeof(cl_double) * params.maxIters * workgroups, av_vels, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
	  printf("Error: Failed to read output array! %s\n", err_code_str(err));
	  exit(1);
	}

    printf("==done==\n");
    rtime = (wtime() - rtime);
    printf("The kernel ran in %lf seconds\n", rtime);

	printf("GPU average velocity: %.12E\n", av_vels[0]);
    printf("Reynolds number:      %.12E\n", calc_reynolds(params, av_vels[0]));

    write_values(params, cells, obstacles, av_vels, workgroups);

	// cleanup, then shutdown
    clReleaseMemObject(cl_cells);
    clReleaseMemObject(cl_dst_cells);
    clReleaseMemObject(cl_obstacles);
    clReleaseMemObject(cl_av_vels);
    clReleaseKernel(accelerate_flow_kernel);
    clReleaseKernel(collision_kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    finalise(&params, &cells, &obstacles, &av_vels);

#ifdef _WIN32
	timeEndPeriod(1);
#endif
    return 0;
}


void initialise_params(t_param* params) {
	FILE *fp;       /* file pointer */
	int retval;     /* to hold return value for checking */

  /* open the parameter file */
  fp = fopen(PARAMFILE,"r");
  if (fp == NULL) {
    die("could not open file input.params",__LINE__,__FILE__);
  }

  /* read in the parameter values */
  retval = fscanf(fp,"%d\n",&(params->nx));
  if(retval != 1) die ("could not read param file: nx",__LINE__,__FILE__);
  retval = fscanf(fp,"%d\n",&(params->ny));
  if(retval != 1) die ("could not read param file: ny",__LINE__,__FILE__);
  retval = fscanf(fp,"%d\n",&(params->maxIters));
  if(retval != 1) die ("could not read param file: maxIters",__LINE__,__FILE__);
  retval = fscanf(fp,"%d\n",&(params->reynolds_dim));
  if(retval != 1) die ("could not read param file: reynolds_dim",__LINE__,__FILE__);
  retval = fscanf(fp,SCANF_FMT,&(params->density));
  if(retval != 1) die ("could not read param file: density",__LINE__,__FILE__);
  retval = fscanf(fp,SCANF_FMT,&(params->accel));
  if(retval != 1) die ("could not read param file: accel",__LINE__,__FILE__);
  retval = fscanf(fp,SCANF_FMT,&(params->omega));
  if(retval != 1) die ("could not read param file: omega",__LINE__,__FILE__);

	/* and close up the file */
	fclose(fp);
}


void initialise_memory(t_param* params, cl_double** cells_ptr, cl_int** obstacles_ptr, cl_double** av_vels_ptr, cl_uint workgroups) {
	FILE *fp;            /* file pointer */
	cl_uint ii,jj;       /* generic counters */
	int xx,yy;           /* generic array indices */
	int blocked;         /* indicates whether a cell is blocked by an obstacle */
	int retval;          /* to hold return value for checking */
	cl_double w0,w1,w2;  /* weighting factors */
	int offset;
	const cl_uint size = params->ny * params->nx;

	/* allocate memory for the two main grid buffers and the obstacle buffer */
	*cells_ptr = (cl_double *) malloc(
			/* cells buffer */
			(sizeof(cl_double) * NSPEEDS * size) +
			/* obstacles buffer */
			(sizeof(cl_int) * size) +
			/* average velocities buffer */
			(sizeof(cl_double) * workgroups * params->maxIters));

	if (*cells_ptr == NULL) {
		die("Could not allocate memory buffers",__LINE__,__FILE__);
	}

	*obstacles_ptr = (cl_int *) &(*cells_ptr)[size * NSPEEDS];
	*av_vels_ptr = (cl_double*) &(*obstacles_ptr)[size];

	/* initialise densities */
	w0 = params->density * 4.0 / 9.0;
	w1 = params->density / 9.0;
	w2 = params->density / 36.0;

	for(ii = 0; ii < params->ny; ii++) {
		for(jj = 0; jj < params->nx; jj++) {
			offset = ii * params->nx + jj;
			/* centre */
			CELL((*cells_ptr), size, offset, 0) = w0;
			/* axis directions */
			CELL((*cells_ptr), size, offset, 1) = w1;
			CELL((*cells_ptr), size, offset, 2) = w1;
			CELL((*cells_ptr), size, offset, 3) = w1;
			CELL((*cells_ptr), size, offset, 4) = w1;
			/* diagonals */
			CELL((*cells_ptr), size, offset, 5) = w2;
			CELL((*cells_ptr), size, offset, 6) = w2;
			CELL((*cells_ptr), size, offset, 7) = w2;
			CELL((*cells_ptr), size, offset, 8) = w2;

			/* set all cells in obstacle array to zero */
			(*obstacles_ptr)[offset] = 0;
		}
	}

	/* open the obstacle data file */
	fp = fopen(OBSTACLEFILE,"r");
	if (fp == NULL) {
		die("could not open file obstacles",__LINE__,__FILE__);
	}

	params->tot_cells = size;

	/* read-in the blocked cells list */
	while( (retval = fscanf(fp,"%d %d %d\n", &xx, &yy, &blocked)) != EOF) {
		/* some checks */
		if ( retval != 3)
			die("expected 3 values per line in obstacle file",__LINE__,__FILE__);
		if ( xx<0 || xx> (int)params->nx-1 )
			die("obstacle x-coord out of range",__LINE__,__FILE__);
		if ( yy<0 || yy> (int)params->ny-1 )
			die("obstacle y-coord out of range",__LINE__,__FILE__);
		if ( blocked != 1 )
			die("obstacle blocked value should be 1",__LINE__,__FILE__);

		/* assign to array */
		(*obstacles_ptr)[yy*params->nx + xx] = (blocked) ? -1 : 0;

		params->tot_cells--;
  }

	/* and close the file */
	fclose(fp);
}

void finalise(const t_param* params, cl_double **cells_ptr, cl_int** obstacles_ptr, cl_double** av_vels_ptr) {
	free(*cells_ptr);
	*cells_ptr = NULL;
	*obstacles_ptr = NULL;
	*av_vels_ptr = NULL;
}

cl_double calc_reynolds(const t_param params, cl_double av_velocity) {
	const cl_double viscosity = 1.0 / 6.0 * (2.0 / params.omega - 1.0);
	return av_velocity * params.reynolds_dim / viscosity;
}


cl_double total_density(const t_param params, cl_double* cells) {
	cl_uint ii, jj, kk;      /* generic counters */
	cl_double total = 0.0;   /* accumulator */
	cl_uint size = params.ny * params.nx;

	for (ii = 0; ii < params.ny; ii++) {
		for (jj = 0; jj < params.nx; jj++) {
			for (kk = 0; kk < NSPEEDS; kk++) {
				total += CELL(cells, size, ii * params.nx + jj, kk);
			}
		}
	}

	return total;
}

void write_values(const t_param params, cl_double* cells, cl_int * obstacles, cl_double* av_vels, cl_uint workgroups) {
	FILE* fp;                        /* file pointer */
	cl_uint ii,jj,kk;                /* generic counters */
	const cl_double c_sq = 1.0/3.0;  /* sq. of speed of sound */
	cl_double local_density;         /* per grid cell sum of densities */
	cl_double pressure;              /* fluid pressure in grid cell */
	cl_double u_x;                   /* x-component of velocity in grid cell */
	cl_double u_y;                   /* y-component of velocity in grid cell */
	const cl_uint size = params.ny * params.nx;

	fp = fopen(FINALSTATEFILE,"w");
	if (fp == NULL) {
		die("could not open file output file",__LINE__,__FILE__);
	}

	for (ii=0;ii<params.ny;ii++) {
		for (jj=0;jj<params.nx;jj++) {
			if (obstacles[ii*params.nx + jj]) {
				/* an occupied cell */
				u_x = u_y = 0.0;
				pressure = params.density * c_sq;
			} else {
				/* no obstacle */
				local_density = 0.0;
				for (kk=0;kk<NSPEEDS;kk++) {
					local_density += CELL(cells, size, ii * params.nx + jj, kk);
				}

				/* compute x velocity component */
				u_x = ((CELL(cells, size, ii * params.nx + jj, 1) +
						CELL(cells, size, ii * params.nx + jj, 5) +
						CELL(cells, size, ii * params.nx + jj, 8)) -
					   (CELL(cells, size, ii * params.nx + jj, 3) +
						CELL(cells, size, ii * params.nx + jj, 6) +
						CELL(cells, size, ii * params.nx + jj, 7)))
						/ local_density;

				/* compute y velocity component */
				u_y = ((CELL(cells, size, ii * params.nx + jj, 2) +
						CELL(cells, size, ii * params.nx + jj, 5) +
						CELL(cells, size, ii * params.nx + jj, 6)) -
					   (CELL(cells, size, ii * params.nx + jj, 4) +
						CELL(cells, size, ii * params.nx + jj, 7) +
						CELL(cells, size, ii * params.nx + jj, 8)))
						/ local_density;

				/* compute pressure */
				pressure = local_density * c_sq;
			}

			/* write to file */
			fprintf(fp,"%d %d %.12E %.12E %.12E %d\n",ii,jj,u_x,u_y,pressure,obstacles[ii*params.nx + jj]);
		}
	}

	fclose(fp);

	fp = fopen(AVVELSFILE,"w");
	if (fp == NULL) {
		die("could not open file output file",__LINE__,__FILE__);
	}
	for (ii=0;ii<params.maxIters;ii++) {
		int iteration_offset = (params.maxIters - 1 - ii);
		fprintf(fp,"%d:\t%.12E\n", ii, av_vels[iteration_offset]);
	}

	fclose(fp);
}


void die(const char* message, const int line, const char *file) {
	fprintf(stderr, "Error at line %d of file %s:\n", line, file);
	fprintf(stderr, "%s\n",message);
	fflush(stderr);
	exit(EXIT_FAILURE);
}


/*
 * A convenience function to prefix messages with MPI related host information.
 */
void debug(const char *format, ...) {
	va_list args;
	printf("[%s] ", "Boltzmann");
	va_start(args, format);
	vprintf(format, args);
	va_end(args);
	fflush(stdout);
}
