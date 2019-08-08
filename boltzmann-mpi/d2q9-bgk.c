/*
 * This software implements a d2q9-bgk lattice boltzmann scheme using MPI.
 * Copyright (c) 2012-2019 Jose Hernandez
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 * ---
 *
 * 'd2' inidates a 2-dimensional grid, and
 * 'q9' indicates 9 velocities per grid cell.
 * 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
 *
 * The 'speeds' in each cell are numbered as follows:
 *
 * 6 2 5
 *  \|/
 * 3-0-1
 *  /|\
 * 7 4 8
 *
 * A 2D grid 'unwrapped' in row major order to give a 1D array:
 *
 *           cols
 *       --- --- ---
 *      | D | E | F |
 * rows  --- --- ---
 *      | A | B | C |
 *       --- --- ---
 *
 *  --- --- --- --- --- ---
 * | A | B | C | D | E | F |
 *  --- --- --- --- --- ---
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <time.h>

#ifndef _WIN32
/* Not available on Microsoft Windows */
#include <sys/time.h>
#endif

#define MIN(a,b)		(((a)<(b))?(a):(b))
#define MAX(a,b)		(((a)>(b))?(a):(b))

#define NSPEEDS         9
#define PARAMFILE       "input.params"
#define OBSTACLEFILE    "obstacles_300x200.dat"
#define FINALSTATEFILE  "final_state%s%s.dat"
#define AVVELSFILE      "av_vels%s%s.dat"

char finalStateFile[128];
char avVelocityFile[128];

/* struct to hold the parameter values */
typedef struct {
    int nx;            /* no. of cells in y-direction */
    int ny;            /* no. of cells in x-direction */
    int maxIters;      /* no. of iterations */
    int reynolds_dim;  /* dimension for Reynolds number */
    double density;    /* density per link */
    double accel;      /* density redistribution */
    double omega;      /* relaxation parameter */

	/* derived values, calculated after loading the input parameters */
	int total_cells;   /* total number of occupied cells (e.g. cells without no obstacles) */
    int tile_ny;      /* no. of cells in x-direction */
    int start_ii;
    int stop_ii;
} t_param;

/* struct to hold the 'speed' values */
typedef struct {
    double speeds[NSPEEDS];
} t_speed;

enum boolean {
    FALSE, TRUE
};

#ifdef DEBUG
#define TRACE(x)			debug x
#define TRACE_RANK(r,x)		do { if (myrank==r) debug x; } while (0)
#else
#define TRACE(x)
#define TRACE_RANK(r,x)
#endif /* DEBUG */

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr, unsigned char** obstacles_ptr,
		double** av_vels_ptr, double **global_av_vels_ptr);

/*
 ** The main calculation methods.
 ** timestep calls, in order, the functions:
 ** accelerate_flow(), propagate(), collision()
 */
int timestep(const t_param params, t_speed* src_cells, t_speed* dst_cells, MPI_Request *requests, MPI_Status *statuses, unsigned char* obstacles);
int accelerate_flow(const t_param params, t_speed* cells, unsigned char* obstacles);
int collision(const t_param params, t_speed* src_cells, t_speed* dst_cells, unsigned char* obstacles, int start, int stop, int increment);
int write_values(const t_param params, t_speed* cells, unsigned char* obstacles, double* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr, unsigned char** obstacles_ptr,
		double** av_vels_ptr, double **global_av_vels_ptr);

/* Sum all the densities in the grid.
 ** The total should remain constant from one timestep to the next. */
double total_density(const t_param params, t_speed* cells);

/* compute average velocity */
double av_velocity(const t_param params, t_speed* cells, unsigned char* obstacles);

/* calculate Reynolds number */
double calc_reynolds(const t_param params, double average_velocity);

/* utility functions */
void die(const char* message, const int line, const char *file);
void debug(const char *format, ...);
int divRoundUp(int totalRowCount, int numProcs);
int isOdd(int n);
int numRowsInTile(int totalRowCount, int numProcs, int rank);

/* MPI variables */
int myrank;               /* 'myrank' of process among it's cohort */ 
int size;               /* size of cohort, i.e. num processes started */
int flag;               /* for checking whether MPI_Init() has been called */
int hostname_strlen;                    /* length of a character array */
char hostname[MPI_MAX_PROCESSOR_NAME];  /* character array to hold hostname running process */
int version;
int subversion;

int do_halo_init(const t_param params, t_speed* cells, MPI_Request *requests);
void debug(const char *format, ...);

/* RESULTS FOR 1 TIMESTEP
==timestep: 0==
av velocity: 5.551908064444E-06
tot density: 5.999999999969E+03
==done==
Reynolds number:        4.108411967689E-03
Elapsed time:           10 (ms)
*/
/* RESULTS FOR 10000 ITERATIONS
Reynolds number:        2.641710388878E+01
Elapsed time:           151320 (ms)
*/

MPI_Request requests[8];
MPI_Status statuses[8];

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[]) {
    t_param params;							/* struct to hold parameter values */
    t_speed* src_cells = NULL;				/* source grid containing fluid densities */
    t_speed* dst_cells = NULL;				/* destination grid containing fluid densities */
    t_speed* temp_swap = NULL;  			/* temporary cell pointer variable used to swap source and destination grid pointers */
    MPI_Request *src_cells_requests_ptr = &requests[0];
    MPI_Request *dst_cells_requests_ptr = &requests[4];
    MPI_Request *tmp_requests_swap = NULL;
    MPI_Status *src_cells_status_ptr = &statuses[0];
    MPI_Status *dst_cells_status_ptr = &statuses[4];
    MPI_Status *tmp_status_swap = NULL;
    unsigned char * obstacles = NULL;     /* grid indicating which cells are blocked */
    double* velocities = NULL;    /* a record of the av. velocity computed for each timestep */
    double* av_vels = NULL;    /* a record of the av. velocity computed for each timestep */
    int ii;                    /* generic counter */
    clock_t tic, toc;          /* check points for reporting processor time used */
	double reynolds_number;
#ifdef DEBUG
	double total_density_value;
#endif
#ifndef _WIN32
    struct timeval tv1,tv2,tv3;
#endif

	if (argc > 1) {
        sprintf(finalStateFile, FINALSTATEFILE, ".", argv[1]);
        sprintf(avVelocityFile, AVVELSFILE, ".", argv[1]);
    } else {
        sprintf(finalStateFile, FINALSTATEFILE, "", "");
        sprintf(avVelocityFile, AVVELSFILE, "", "");
    }

	/* initialise our MPI environment */
	MPI_Init( &argc, &argv );

	/* check whether the initialisation was successful */
	MPI_Initialized(&flag);
	if ( flag != TRUE ) {
		MPI_Abort(MPI_COMM_WORLD,EXIT_FAILURE);
	}

	/* determine the hostname */
	MPI_Get_processor_name(hostname,&hostname_strlen);

	/* 
	** determine the SIZE of the group of processes associated with
	** the 'communicator'.  MPI_COMM_WORLD is the default communicator
	** consisting of all the processes in the launched MPI 'job'
	*/
	MPI_Comm_size( MPI_COMM_WORLD, &size );

	/* determine the RANK of the current process [0:SIZE-1] */
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Get_version(&version, &subversion);

	TRACE_RANK(0, ("sizeof(double):        %lu\n", sizeof(double)));
	TRACE_RANK(0, ("sizeof(int):           %lu\n", sizeof(int)));
	TRACE_RANK(0, ("sizeof(int *):         %lu\n", sizeof(int *)));
	TRACE_RANK(0, ("sizeof(unsigned char): %lu\n", sizeof(unsigned char)));

	/* 
	** make use of these values in our print statement
	** note that we are assuming that all processes can
	** write to the screen
	*/
	if (0 == myrank) {
		debug("Starting Boltzmann MPI master node\n");
		debug("MPI library version %d.%d\n", version, subversion);
		debug("Cohort size: %d\n", size);
	} else {
		debug("Starting Boltzmann MPI client node %d\n", myrank);
		debug("MPI library version %d.%d\n", version, subversion);
		debug("Cohort size: %d\n", size);
	}

	/* initialise our data structures and load values from file */
    initialise(&params, &src_cells, &dst_cells, &obstacles, &velocities, &av_vels);

	/* iterate for maxIters time steps */
    tic = clock();
#ifndef _WIN32
    gettimeofday(&tv1, NULL);
#endif

    do_halo_init(params, src_cells, src_cells_requests_ptr);
    do_halo_init(params, dst_cells, dst_cells_requests_ptr);

    for (ii = 0; ii < params.maxIters; ii++) {
        timestep(params, src_cells, dst_cells, src_cells_requests_ptr, src_cells_status_ptr, obstacles);
        velocities[ii] = av_velocity(params, dst_cells, obstacles);

		temp_swap = src_cells;
        src_cells = dst_cells;
        dst_cells = temp_swap;

        tmp_requests_swap = src_cells_requests_ptr;
        src_cells_requests_ptr = dst_cells_requests_ptr;
        dst_cells_requests_ptr = tmp_requests_swap;

        tmp_status_swap = src_cells_status_ptr;
        src_cells_status_ptr = dst_cells_status_ptr;
        dst_cells_status_ptr = tmp_status_swap;
#ifdef DEBUG
		total_density_value = total_density(params, src_cells);
		TRACE(("==timestep: %d==\n", ii));
		TRACE(("partial velocity: %.12E\n", velocities[ii]));
		TRACE(("tot density: %.12E\n", total_density_value));
#endif
    }

    for (ii = 0; ii < 8; ii++) {
    	MPI_Request_free(&requests[ii]);
    }

    // Obtain the sum of all the velocities across the cohort
	TRACE(("Beginning average velocity MPI reduce\n"));
	MPI_Reduce(velocities, av_vels, params.maxIters, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	TRACE(("... ending average velocity MPI reduce\n"));

	// compute all the average velocities
	for (ii = 0; ii < params.maxIters; ii++) {
		av_vels[ii] /= params.total_cells;
		TRACE_RANK(0, ("av velocity: %.12E\n", av_vels[ii]));
    }

#ifndef _WIN32
    gettimeofday(&tv2, NULL);
    timersub(&tv2, &tv1, &tv3);
#endif
    toc = clock();

    /* write final values and free memory */
	reynolds_number = calc_reynolds(params, av_vels[params.maxIters - 1]);
	if (0 == myrank) {
	    debug("==done==\n");
		debug("Reynolds number:\t%.12E\n", reynolds_number);
		debug("Elapsed CPU time:\t%ld (ms)\n", (toc - tic) / (CLOCKS_PER_SEC / 1000));
#ifndef _WIN32
	    debug("Elapsed wall time:\t%ld (ms)\n", (tv3.tv_sec * 1000) + (tv3.tv_usec / 1000));
#endif
	}

	write_values(params, src_cells, obstacles, av_vels);
	finalise(&params, &src_cells, &dst_cells, &obstacles, &velocities, &av_vels);

	/* finalise the MPI environment */
	MPI_Finalize();

    return EXIT_SUCCESS;
}

int timestep(const t_param params, t_speed* src_cells, t_speed* dst_cells, MPI_Request *requests, MPI_Status *statuses, unsigned char* obstacles) {
    accelerate_flow(params, src_cells, obstacles);
	TRACE(("Starting halo exchange\n"));
	MPI_Startall(4, requests);
	collision(params, src_cells, dst_cells, obstacles, params.start_ii + 1, params.stop_ii - 1, 1);
	TRACE(("Waiting for halo exchange to complete\n"));
	MPI_Waitall(4, requests, statuses);
	collision(params, src_cells, dst_cells, obstacles, params.start_ii, params.stop_ii, params.stop_ii - params.start_ii - 1);
    return EXIT_SUCCESS;
}

int accelerate_flow(const t_param params, t_speed * cells, unsigned char* obstacles) {
    int ii, offset; /* generic counters */
    double * speeds;

    /* compute weighting factors */
    const double w1 = params.density * params.accel / 9.0;
    const double w2 = params.density * params.accel / 36.0;

    /* modify the first column of the grid */
    for (ii = params.start_ii; ii < params.stop_ii; ii++) {
        offset = ii * params.nx;
        speeds = cells[offset].speeds;
        /* if the cell is not occupied and we don't send a density negative */
        if (!obstacles[offset] && (speeds[3] - w1) > 0.0 && (speeds[6] - w2) > 0.0 && (speeds[7] - w2) > 0.0) {
            /* increase 'east-side' densities */
            speeds[1] += w1;
            speeds[5] += w2;
            speeds[8] += w2;
            /* decrease 'west-side' densities */
            speeds[3] -= w1;
            speeds[6] -= w2;
            speeds[7] -= w2;
        }
    }

    return EXIT_SUCCESS;
}

int collision(const t_param params, t_speed* src_cells, t_speed* dst_cells, unsigned char* obstacles, int start, int stop, int increment) {
    int ii, jj, kk, offset;            /* generic counters */
    const double c_sq = 1.0 / 3.0;     /* square of speed of sound */
    const double w0 = 4.0 / 9.0;       /* weighting factor */
    const double w1 = 1.0 / 9.0;       /* weighting factor */
    const double w2 = 1.0 / 36.0;      /* weighting factor */
    double u_x, u_y;                   /* av. velocities in x and y directions */
    double u[NSPEEDS];                 /* directional velocities */
    double d_equ[NSPEEDS];             /* equilibrium densities */
    double u_sq;                       /* squared velocity */
    double local_density;              /* sum of densities in a particular cell */
    int x_e, x_w, y_n, y_s;            /* indices of neighbouring cells */
    double speeds[NSPEEDS];
    double *dst_speeds;

    /* divide the workload amongst the number of nodes in the cohort so that each myrank takes on just one sub-section of
     * the array.
     */
	TRACE(("processing ii range [%03d, %03d) in steps of size %d\n", start, stop, increment));

	/* loop over the cells in the grid.
     * NB: the collision step is called after the propagate step and so values of interest are in the scratch-space grid
     */
    for (ii = start; ii < stop; ii += increment) {
        for (jj = 0; jj < params.nx; jj++) {
            offset = ii * params.nx + jj;
            dst_speeds = dst_cells[offset].speeds;

            /* PROPAGATE: determine indices of axis-direction neighbours respecting periodic boundary conditions (wrap around) */
            y_s = ii + 1;
            x_w = (jj + 1) % params.nx;
            y_n = ii - 1;
            x_e = (jj == 0) ? (jj + params.nx - 1) : (jj - 1);

            /* if the cell contains an obstacle */
            if (obstacles[ii * params.nx + jj]) {
                /* PROPAGATE ANB REBOUND: propagate and mirror densities from neighbouring cells into the current cell. */
                dst_speeds[0] = src_cells[ ii * params.nx + jj ].speeds[0];  /* central cell, no movement */
                dst_speeds[1] = src_cells[ ii * params.nx + x_w].speeds[3];  /* west */
                dst_speeds[2] = src_cells[y_s * params.nx + jj ].speeds[4];  /* south */
                dst_speeds[3] = src_cells[ ii * params.nx + x_e].speeds[1];  /* east */
                dst_speeds[4] = src_cells[y_n * params.nx + jj ].speeds[2];  /* north */
                dst_speeds[5] = src_cells[y_s * params.nx + x_w].speeds[7];  /* south-west */
                dst_speeds[6] = src_cells[y_s * params.nx + x_e].speeds[8];  /* south-east */
                dst_speeds[7] = src_cells[y_n * params.nx + x_e].speeds[5];  /* north-east */
                dst_speeds[8] = src_cells[y_n * params.nx + x_w].speeds[6];  /* north-west */
            } else {
                /* PROPAGATE: propagate densities from neighbouring cells into the current cells, following appropriate directions of
                 * travel and writing into a temporary buffer.
                 */
                speeds[0] = src_cells[ ii * params.nx + jj ].speeds[0];  /* central cell, no movement */
                speeds[1] = src_cells[ ii * params.nx + x_e].speeds[1];  /* east */
                speeds[2] = src_cells[y_n * params.nx + jj ].speeds[2];  /* north */
                speeds[3] = src_cells[ ii * params.nx + x_w].speeds[3];  /* west */
                speeds[4] = src_cells[y_s * params.nx + jj ].speeds[4];  /* south */
                speeds[5] = src_cells[y_n * params.nx + x_e].speeds[5];  /* north-east */
                speeds[6] = src_cells[y_n * params.nx + x_w].speeds[6];  /* north-west */
                speeds[7] = src_cells[y_s * params.nx + x_w].speeds[7];  /* south-west */
                speeds[8] = src_cells[y_s * params.nx + x_e].speeds[8];  /* south-east */

                /* COLLISION */
                /* compute local density total */
                local_density = speeds[0] + speeds[1] + speeds[2] + speeds[3] + speeds[4] + speeds[5] + speeds[6] + speeds[7] + speeds[8];
                /* compute x velocity component */
                u_x = (speeds[1] + speeds[5] + speeds[8] - (speeds[3] + speeds[6] + speeds[7])) / local_density;
                /* compute y velocity component */
                u_y = (speeds[2] + speeds[5] + speeds[6] - (speeds[4] + speeds[7] + speeds[8])) / local_density;
                /* velocity squared */
                u_sq = u_x * u_x + u_y * u_y;
                /* directional velocity components */
                u[1] =  u_x;       /* east */
                u[2] =  u_y;       /* north */
                u[5] =  u_x + u_y; /* north-east */
                u[6] = -u_x + u_y; /* north-west */
                /* equilibrium densities */
                /* zero velocity density: weight w0 */
                d_equ[0] = w0 * local_density * (1.0 - u_sq / (2.0 * c_sq));
                /* axis speeds: weight w1 */
                d_equ[1] = w1 * local_density * (1.0 + u[1] / c_sq + (u[1] * u[1]) / (2.0 * c_sq * c_sq) - u_sq / (2.0 * c_sq));
                d_equ[2] = w1 * local_density * (1.0 + u[2] / c_sq + (u[2] * u[2]) / (2.0 * c_sq * c_sq) - u_sq / (2.0 * c_sq));
                d_equ[3] = w1 * local_density * (1.0 - u[1] / c_sq + (u[1] * u[1]) / (2.0 * c_sq * c_sq) - u_sq / (2.0 * c_sq));
                d_equ[4] = w1 * local_density * (1.0 - u[2] / c_sq + (u[2] * u[2]) / (2.0 * c_sq * c_sq) - u_sq / (2.0 * c_sq));
                /* diagonal speeds: weight w2 */
                d_equ[5] = w2 * local_density * (1.0 + u[5] / c_sq + (u[5] * u[5]) / (2.0 * c_sq * c_sq) - u_sq / (2.0 * c_sq));
                d_equ[6] = w2 * local_density * (1.0 + u[6] / c_sq + (u[6] * u[6]) / (2.0 * c_sq * c_sq) - u_sq / (2.0 * c_sq));
                d_equ[7] = w2 * local_density * (1.0 - u[5] / c_sq + (u[5] * u[5]) / (2.0 * c_sq * c_sq) - u_sq / (2.0 * c_sq));
                d_equ[8] = w2 * local_density * (1.0 - u[6] / c_sq + (u[6] * u[6]) / (2.0 * c_sq * c_sq) - u_sq / (2.0 * c_sq));
                /* relaxation step */
                for (kk = 0; kk < NSPEEDS; kk++) {
                    dst_speeds[kk] = (speeds[kk] + params.omega * (d_equ[kk] - speeds[kk]));
                }
            }
        }
    }

    return EXIT_SUCCESS;
}

void send_halo_init(int Id, t_speed* const cells, int offset, int xchgbuffersize, int destination, int tag, MPI_Request *request) {
	TRACE(("%d Initialising send request from process '%d' to process %d; cell buffer offset = %d\n", Id, myrank, destination, offset));
	MPI_Send_init(&cells[offset], xchgbuffersize, MPI_UNSIGNED_CHAR, destination, tag, MPI_COMM_WORLD, request);
	TRACE(("%d Initialised send request from process '%d' to process %d\n", Id, myrank, destination));
}

void receive_halo_init(int Id, t_speed* const cells, int offset, int xchgbuffersize, int source, int tag, MPI_Request *request) {
	TRACE(("%d Initialising receive request in process '%d' from process '%d'; cell buffer offset = %d\n", Id, myrank, source, offset));
	MPI_Recv_init(&cells[offset], xchgbuffersize, MPI_UNSIGNED_CHAR, source, tag, MPI_COMM_WORLD, request);
	TRACE(("%d Initialising receive request in process '%d' from process '%d'\n", Id, myrank, source));
}

int do_halo_init(const t_param params, t_speed* cells, MPI_Request *requests) {
	const static int left = 0;
	const static int right = 0;
	int i, source, destination, offset;
	int xchgbuffersize = params.nx * sizeof(t_speed);

	/* each process has to send and receive one row, we'll split sending and receiving between odd and even processes
	 * to avoid deadlock.
	 */
	for (i = 0; i < 2; i++) {
		if ((myrank % 2) == i) {
			source = myrank;

			destination = (myrank > 0) ? ((myrank - 1) % size) : (size - 1);
			offset = 0;
			send_halo_init(0, cells, offset, xchgbuffersize, destination, left, &requests[0]);

			destination = (myrank + 1) % size;
			offset = (params.stop_ii - 1) * params.nx;
			send_halo_init(1, cells, offset, xchgbuffersize, destination, right, &requests[1]);
		} else {
			destination = myrank;

			source = (myrank + 1) % size;
			offset = params.stop_ii * params.nx;
			receive_halo_init(2, cells, offset, xchgbuffersize, source, left, &requests[2]);

			source = (myrank > 0) ? ((myrank - 1) % size) : (size - 1);
			offset = -params.nx;
			receive_halo_init(3, cells, offset, xchgbuffersize, source, right, &requests[3]);
		}
	}

	return EXIT_SUCCESS;
}

int initialise(t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr, unsigned char** obstacles_ptr,
		double** av_vels_ptr, double **global_av_vels_ptr) {
    FILE *fp;          /* file pointer */
    int ii, jj;        /* generic counters */
    int xx, yy;        /* generic array indices */
    int blocked;       /* indicates whether a cell is blocked by an obstacle */
    int retval;        /* to hold return value for checking */
    double w0, w1, w2; /* weighting factors */
	int obstacle_array_size;  /* size of the obstacle array in bytes */
	int cell_array_size;

	if (0 == myrank) {
		/* open the parameter file */
		fp = fopen(PARAMFILE, "r");
		if (fp == NULL) {
			die("could not open file input.params", __LINE__, __FILE__);
		}

		/* read in the parameter values */
		retval = fscanf(fp, "%d\n", &(params->nx));
		if (retval != 1)
			die("could not read param file: nx", __LINE__, __FILE__);
		retval = fscanf(fp, "%d\n", &(params->ny));
		if (retval != 1)
			die("could not read param file: ny", __LINE__, __FILE__);
		retval = fscanf(fp, "%d\n", &(params->maxIters));
		if (retval != 1)
			die("could not read param file: maxIters", __LINE__, __FILE__);
		retval = fscanf(fp, "%d\n", &(params->reynolds_dim));
		if (retval != 1)
			die("could not read param file: reynolds_dim", __LINE__, __FILE__);
		retval = fscanf(fp, "%lf\n", &(params->density));
		if (retval != 1)
			die("could not read param file: density", __LINE__, __FILE__);
		retval = fscanf(fp, "%lf\n", &(params->accel));
		if (retval != 1)
			die("could not read param file: accel", __LINE__, __FILE__);
		retval = fscanf(fp, "%lf\n", &(params->omega));
		if (retval != 1)
			die("could not read param file: omega", __LINE__, __FILE__);

		/* and close up the file */
		fclose(fp);
	}

	/* If the number of rows does not divide nicely into the number of processors we'll increment the tile row count by
	 * one and make sure that the last process/rank adjusts its internal count not to exceed the real total row count.
	 */
	params->tile_ny = divRoundUp(params->ny, size);

	TRACE(((0 == myrank) ? "Broadcasting input parameters ...\n" : "Awaiting reception of input parameters ...\n"));
	MPI_Bcast(params, sizeof(*params), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
	TRACE(((0 == myrank) ? "... input parameter broadcast complete\n": "... input parameter reception complete\n"));

	TRACE(("Parameter nx = %d\n", params->nx));
	TRACE(("Parameter ny = %d\n", params->ny));
	TRACE(("Parameter max. iterations = %d\n", params->maxIters));
	TRACE(("Parameter Reynolds dimension = %d\n", params->reynolds_dim));
	TRACE(("Parameter density = %lf\n", params->density));
	TRACE(("Parameter acceleration = %lf\n", params->accel));
	TRACE(("Parameter omega %lf\n", params->omega));

    params->start_ii = 0;
    params->stop_ii = numRowsInTile(params->ny, size, myrank);
    TRACE(("start %d, stop %d\n", params->start_ii, params->stop_ii));

	/*
     * Allocate memory.
     *
     * Remember C is pass-by-value, so we need to pass pointers into the initialise function.
     *
     * NB we are allocating a 1D array, so that the memory will be contiguous.  We still want to index this memory as if it were a (row
     * major ordered) 2D array, however.  We will perform some arithmetic using the row and column coordinates, inside the square brackets,
     * when we want to access elements of this array.
     *
     * Note also that we are using a structure to hold an array of 'speeds'.  We will allocate a 1D array of these structs.
     */

    /* main grid. size is given by the local block height plus the row before and after multiplied by the row length. */
    TRACE(("Allocating source cell array\n"));
    cell_array_size = sizeof(t_speed) * ((params->tile_ny + 2) * params->nx);
    *cells_ptr = (t_speed*) malloc(cell_array_size);
    if (*cells_ptr == NULL)
        die("cannot allocate memory for cells", __LINE__, __FILE__);
	memset(*cells_ptr, 0, cell_array_size);
	*cells_ptr = &(*cells_ptr)[params->nx];

    /* 'helper' grid, used as scratch space, its size is given by the local block height plus the row before and after
     * multiplied by the row length.
     */
    TRACE(("Allocating temporary cell array\n"));
    *tmp_cells_ptr = (t_speed*) malloc(cell_array_size);
    if (*tmp_cells_ptr == NULL)
        die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);
	memset(*tmp_cells_ptr, 0, cell_array_size);
	*tmp_cells_ptr = &(*tmp_cells_ptr)[params->nx];

	/* the map of obstacles */
    TRACE(("Allocating obstacle array\n"));
	obstacle_array_size = sizeof(unsigned char) * (params->ny * params->nx);
	*obstacles_ptr = (unsigned char *) malloc(obstacle_array_size);
    if (*obstacles_ptr == NULL)
        die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

    /* initialise densities */
    w0 = params->density * 4.0 / 9.0;
    w1 = params->density / 9.0;
    w2 = params->density / 36.0;

    TRACE(("Initialising cell array\n"));
    for (ii = 0; ii < params->tile_ny; ii++) {
        for (jj = 0; jj < params->nx; jj++) {
            /* centre */
            (*cells_ptr)[ii * params->nx + jj].speeds[0] = w0;
            /* axis directions */
            (*cells_ptr)[ii * params->nx + jj].speeds[1] = w1;
            (*cells_ptr)[ii * params->nx + jj].speeds[2] = w1;
            (*cells_ptr)[ii * params->nx + jj].speeds[3] = w1;
            (*cells_ptr)[ii * params->nx + jj].speeds[4] = w1;
            /* diagonals */
            (*cells_ptr)[ii * params->nx + jj].speeds[5] = w2;
            (*cells_ptr)[ii * params->nx + jj].speeds[6] = w2;
            (*cells_ptr)[ii * params->nx + jj].speeds[7] = w2;
            (*cells_ptr)[ii * params->nx + jj].speeds[8] = w2;
        }
    }

	if (0 == myrank) {
		/* first set all cells in obstacle array to zero */
		memset(*obstacles_ptr, 0, obstacle_array_size);

		/* open the obstacle data file */
		fp = fopen(OBSTACLEFILE, "r");
		if (fp == NULL) {
			die("could not open file obstacles", __LINE__, __FILE__);
		}

		params->total_cells = params->ny * params->nx;

		/* read-in the blocked cells list */
		while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF) {
			/* some checks */
			if (retval != 3)
				die("expected 3 values per line in obstacle file", __LINE__, __FILE__);
			if (xx < 0 || xx > params->nx - 1)
				die("obstacle x-coord out of range", __LINE__, __FILE__);
			if (yy < 0 || yy > params->ny - 1)
				die("obstacle y-coord out of range", __LINE__, __FILE__);
			if (blocked != 1)
				die("obstacle blocked value should be 1", __LINE__, __FILE__);
			/* assign to array */
			(*obstacles_ptr)[yy * params->nx + xx] = blocked;
			if (0 != blocked) {
				params->total_cells--;
			}
		}

		/* and close the file */
		fclose(fp);
	}

	TRACE(((0 == myrank) ? "Broadcasting obstacle array ...\n" : "Awaiting reception of obstacle array ...\n"));
	MPI_Bcast(*obstacles_ptr, obstacle_array_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
	TRACE(((0 == myrank) ? "... obstacle array broadcast complete\n" : "... obstacle array reception complete\n"));

	{
		obstacle_array_size = sizeof(unsigned char) * (params->tile_ny * params->nx);
	    unsigned char* small_obstacles_ptr = (unsigned char *) malloc(obstacle_array_size);
	    if (small_obstacles_ptr == NULL) {
	        die("cannot allocate column memory for obstacles", __LINE__, __FILE__);
	    }
		memcpy(small_obstacles_ptr, &(*obstacles_ptr)[myrank * params->tile_ny * params->nx], (params->stop_ii) * params->nx);
        free(*obstacles_ptr);
    	*obstacles_ptr = small_obstacles_ptr;
	}

	/* allocate space to hold a record of the avarage velocities computed at each timestep */
    *av_vels_ptr = (double*) malloc(sizeof(double) * params->maxIters);
    *global_av_vels_ptr = (double*) malloc(sizeof(double) * params->maxIters);

    return EXIT_SUCCESS;
}

int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr, unsigned char** obstacles_ptr,
		double** av_vels_ptr, double **global_av_vels_ptr) {
    /*
     ** free up allocated memory
     */
    free(&(*cells_ptr)[-params->nx]);
    *cells_ptr = NULL;

    free(&(*tmp_cells_ptr)[-params->nx]);
    *tmp_cells_ptr = NULL;

    free(*obstacles_ptr);
    *obstacles_ptr = NULL;

    free(*av_vels_ptr);
    *av_vels_ptr = NULL;

    free(*global_av_vels_ptr);
    *global_av_vels_ptr = NULL;

    return EXIT_SUCCESS;
}

double av_velocity(const t_param params, t_speed* cells, unsigned char* obstacles) {
    int ii, jj, offset;             /* generic counters */
    double local_density;           /* total density in cell */
    double tot_u_x = 0.0;           /* accumulated x-components of velocity */
    double *speeds;                 /* directional speeds */

	TRACE(("processing velocity ii range [%03d, %03d)\n", params.start_ii, params.stop_ii));

    /* Loop over all non-blocked cells.  Please note that at this point we only compute the average velocity for the
	 * block we're working with.  The overall averages will be calculated at the end of the simulation.
	 */
    for (ii = params.start_ii; ii < params.stop_ii; ii++) {
        for (jj = 0; jj < params.nx; jj++) {
            offset = ii * params.nx + jj;
            /* ignore occupied cells */
            if (!obstacles[offset]) {
                speeds = cells[offset].speeds;
                /* local density total */
                local_density = speeds[0] + speeds[1] + speeds[2] + speeds[3] + speeds[4] + speeds[5] + speeds[6] + speeds[7] + speeds[8];
                /* x-component of velocity */
                tot_u_x += (speeds[1] + speeds[5] + speeds[8] - (speeds[3] + speeds[6] + speeds[7])) / local_density;
            }
        }
    }

	return tot_u_x;
}

double calc_reynolds(const t_param params, double average_velocity) {
    const double viscosity = 1.0 / 6.0 * (2.0 / params.omega - 1.0);
    return average_velocity * params.reynolds_dim / viscosity;
}

double total_density(const t_param params, t_speed* cells) {
    int ii, jj, kk;        /* generic counters */
    double partial = 0.0;  /* partial accumulator */
	double total = 0.0;    /* accumulator */

	TRACE(("processing total density ii range [%03d, %03d)\n", params.start_ii, params.stop_ii));

    for (ii = params.start_ii; ii < params.stop_ii; ii++) {
        for (jj = 0; jj < params.nx; jj++) {
            for (kk = 0; kk < NSPEEDS; kk++) {
                partial += cells[ii * params.nx + jj].speeds[kk];
            }
        }
    }

	TRACE(("Beginning total density MPI reduce with partial density value %lf...\n", partial));
	MPI_Reduce(&partial, &total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	TRACE(("... ending total density MPI reduce\n"));

    return total;
}

int write_values(const t_param params, t_speed* cells, unsigned char* obstacles, double* av_vels) {
    FILE* fp; /* file pointer */
    int i, ii, jj, offset; /* generic counters */
    const double c_sq = 1.0 / 3.0; /* sq. of speed of sound */
    double local_density; /* per grid cell sum of densities */
    double pressure; /* fluid pressure in grid cell */
    double u_x; /* x-component of velocity in grid cell */
    double u_y; /* y-component of velocity in grid cell */
    double * speeds;
	const int cellbuffersize = params.tile_ny * params.nx * sizeof(t_speed);
	const int obstaclebuffersize = params.tile_ny * params.nx * sizeof(unsigned char);

    if (0 == myrank) {
    	MPI_Status status;

    	fp = fopen(finalStateFile, "w");

    	if (fp == NULL) {
			die("could not open file output file", __LINE__, __FILE__);
		}

		for (i = 0; i < size; i++) {
			TRACE(("Writing process %d to file\n", i));
			if (i > 0) {
				TRACE(("Awaiting buffer from process %d\n", i));
				MPI_Recv(cells, cellbuffersize, MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD, &status);
				MPI_Recv(obstacles, obstaclebuffersize, MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD, &status);
				TRACE(("Received buffer from process %d\n", i));
			}
			TRACE(("Range [%d, %d)\n", i * params.tile_ny, MIN(((i+1) * params.tile_ny), (params.ny))));
			for (ii = i * params.tile_ny; ii < MIN(((i+1) * params.tile_ny), (params.ny)); ii++) {
				for (jj = 0; jj < params.nx; jj++) {
					offset = i * params.nx + jj;
					/* an occupied cell */
					if (obstacles[offset]) {
						u_x = u_y = 0.0;
						pressure = params.density * c_sq;
					} else /* no obstacle */{
						speeds = cells[offset].speeds;
						local_density = speeds[0] + speeds[1] + speeds[2] + speeds[3] + speeds[4] + speeds[5]
								+ speeds[6] + speeds[7] + speeds[8];
						/* compute x velocity component */
						u_x = (speeds[1] + speeds[5] + speeds[8] - (speeds[3] + speeds[6] + speeds[7])) / local_density;
						/* compute y velocity component */
						u_y = (speeds[2] + speeds[5] + speeds[6] - (speeds[4] + speeds[7] + speeds[8])) / local_density;
						/* compute pressure */
						pressure = local_density * c_sq;
					}
					/* write to file */
					fprintf(fp, "%d %d %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, pressure, obstacles[offset]);
				}
			}
			fflush(fp);
		}

		fclose(fp);

		fp = fopen(avVelocityFile, "w");
		if (fp == NULL) {
			die("could not open file output file", __LINE__, __FILE__);
		}
		for (ii = 0; ii < params.maxIters; ii++) {
			fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
		}

		fclose(fp);
	} else {
		TRACE(("Sending buffer from process %d to root\n", myrank));
		MPI_Send(cells, cellbuffersize, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);
		MPI_Send(obstacles, obstaclebuffersize, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);
		TRACE(("Sent buffer from process %d to root\n", myrank));
	}

    return EXIT_SUCCESS;
}

void die(const char* message, const int line, const char *file) {
    fprintf(stderr, "Error at line %d of file %s:\n", line, file);
    fprintf(stderr, "%s\n", message);
    fflush(stderr);
    exit(EXIT_FAILURE);
}

/*
 * A convenience function to prefix messages with MPI related host information. 
 */
void debug(const char *format, ...) {
	va_list args;
	printf("[host %s: process %02d of %02d] ", hostname, myrank, size);
	va_start(args, format);
	vprintf(format, args);
	va_end(args);
	fflush(stdout);
}

int divRoundUp(int totalRowCount, int numProcs) {
	return (totalRowCount + numProcs - 1) / numProcs;
}

int isOdd(int n) {
	return n % 2 != 0;
}

int numRowsInTile(int totalRowCount, int numProcs, int rank) {
	int q = divRoundUp(totalRowCount, numProcs);
	return  MIN(totalRowCount, q * (rank + 1)) - (q * rank);
}
