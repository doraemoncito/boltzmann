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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef _WIN32
/* Not available on Microsoft Windows */
#include <sys/time.h>
#endif

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

    /* pre-computed offsets buffer and pointers */
    int *lookup_buffer;
    int *lookup_y_s;
    int *lookup_y_n;
    int *lookup_x_w;
    int *lookup_x_e;
} t_param;

/* struct to hold the 'speed' values */
typedef struct {
    double speeds[NSPEEDS];
} t_speed;

enum boolean {
    FALSE, TRUE
};

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr, unsigned char** obstacles_ptr, double** av_vels_ptr);

/* 
 ** The main calculation methods.
 ** timestep calls, in order, the functions:
 ** accelerate_flow(), propagate(), collision()
 */
int timestep(const t_param params, t_speed* src_cells, t_speed* dst_cells, unsigned char* obstacles);
int accelerate_flow(const t_param params, t_speed* cells, unsigned char* obstacles);
int collision(const t_param params, t_speed* dst_cells, t_speed* src_cells, unsigned char* obstacles);
int write_values(const t_param params, t_speed* cells, unsigned char* obstacles, double* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr, unsigned char** obstacles_ptr, double** av_vels_ptr);

/* Sum all the densities in the grid.
 ** The total should remain constant from one timestep to the next. */
double total_density(const t_param params, t_speed* cells);

/* compute average velocity */
double av_velocity(const t_param params, t_speed* cells, unsigned char* obstacles);

/* calculate Reynolds number */
double calc_reynolds(const t_param params, t_speed* cells, unsigned char* obstacles);

/* utility functions */
void die(const char* message, const int line, const char *file);

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[]) {
    t_param params;            /* struct to hold parameter values */
    t_speed* src_cells = NULL; /* source grid containing fluid densities */
    t_speed* dst_cells = NULL; /* destination grid containing fluid densities */
    t_speed* temp_swap = NULL;  /* temporary cell pointer variable used to swap source and destination grid pointers */
    unsigned char * obstacles = NULL;     /* grid indicating which cells are blocked */
    double* av_vels = NULL;    /* a record of the av. velocity computed for each timestep */
    int ii;                    /* generic counter */
    clock_t tic, toc;          /* check points for reporting processor time used */

    printf("d2q9-bgk lattice Boltzmann scheme: optimised serial code\n");

#ifndef _WIN32
    struct timeval tv1,tv2,tv3;
#endif
#ifdef DEBUG
    printf("sizeof(double):        %lu\n", sizeof(double));
    printf("sizeof(int):           %lu\n", sizeof(int));
    printf("sizeof(int *):         %lu\n", sizeof(int *));
    printf("sizeof(unsigned char): %lu\n", sizeof(unsigned char));
#endif
    if (argc > 1) {
        sprintf(finalStateFile, FINALSTATEFILE, ".", argv[1]);
        sprintf(avVelocityFile, AVVELSFILE, ".", argv[1]);
    } else {
        sprintf(finalStateFile, FINALSTATEFILE, "", "");
        sprintf(avVelocityFile, AVVELSFILE, "", "");
    }

    /* initialise our data structures and load values from file */
    initialise(&params, &src_cells, &dst_cells, &obstacles, &av_vels);

    /* iterate for maxIters timesteps */
    tic = clock();
#ifndef _WIN32
    gettimeofday(&tv1, NULL);
#endif

    for (ii = 0; ii < params.maxIters; ii++) {
        timestep(params, src_cells, dst_cells, obstacles);
        temp_swap = src_cells;
        src_cells = dst_cells;
        dst_cells = temp_swap;
        av_vels[ii] = av_velocity(params, src_cells, obstacles);
#ifdef DEBUG
        printf("==timestep: %d==\n",ii);
        printf("av velocity: %.12E\n", av_vels[ii]);
        printf("tot density: %.12E\n",total_density(params,src_cells));
#endif
    }

#ifndef _WIN32
    gettimeofday(&tv2, NULL);
    timersub(&tv2, &tv1, &tv3);
#endif
    toc = clock();

    /* write final values and free memory */
    printf("==done==\n");
    printf("Reynolds number:\t%.12E\n", calc_reynolds(params, src_cells, obstacles));
    printf("Elapsed CPU time: \t%ld (ms)\n", (toc - tic) / (CLOCKS_PER_SEC / 1000));
#ifndef _WIN32
    printf("Elapsed wall time:\t%ld (ms)\n", (tv3.tv_sec * 1000) + (tv3.tv_usec / 1000));
#endif
    write_values(params, src_cells, obstacles, av_vels);
    finalise(&params, &src_cells, &dst_cells, &obstacles, &av_vels);

    return EXIT_SUCCESS;
}

int timestep(const t_param params, t_speed* src_cells, t_speed* dst_cells, unsigned char* obstacles) {
    accelerate_flow(params, src_cells, obstacles);
    collision(params, src_cells, dst_cells, obstacles);
    return EXIT_SUCCESS;
}

int accelerate_flow(const t_param params, t_speed * cells, unsigned char* obstacles) {
    int ii, offset; /* generic counters */
    double * speeds;

    /* compute weighting factors */
    const double w1 = params.density * params.accel / 9.0;
    const double w2 = params.density * params.accel / 36.0;

    /* modify the first column of the grid */
    for (ii = 0; ii < params.ny; ii++) {
        offset = ii * params.nx /* + jj (where jj=0) */;
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

/*
 * A combined and rebound and collision function.
 */
int collision(const t_param params, t_speed* src_cells, t_speed* dst_cells, unsigned char* obstacles) {
    int ii, jj, kk, offset;            /* generic counters */
#ifdef BOLTZMANN_ACCURATE
    const double c_sq = 1.0 / 3.0;     /* square of speed of sound */
#endif
    const double w0 = 4.0 / 9.0;       /* weighting factor */
    const double w1 = 1.0 / 9.0;       /* weighting factor */
    const double w2 = 1.0 / 36.0;      /* weighting factor */
    double u_x, u_y;                   /* av. velocities in x and y directions */
    double u[NSPEEDS];                 /* directional velocities */
    double d_equ[NSPEEDS];             /* equilibrium densities */
    double u_sq;                       /* squared velocity */
    double local_density;              /* sum of densities in a particular cell */
    int x_e, x_w, y_n, y_s;            /* indices of neighbouring cells */
    double *dst_speeds;
    double speeds[NSPEEDS];

    /* loop over the cells in the grid.
     * NB: the collision step is called after the propagate step and so values of interest are in the scratch-space grid
     */
    for (ii = 0; ii < params.ny; ii++) {
        offset = ii * params.nx;
        y_s = params.lookup_y_s[ii];
        y_n = params.lookup_y_n[ii];
        for (jj = 0; jj < params.nx; jj++) {
            dst_speeds = dst_cells[offset++].speeds;

            /* PROPAGATE: determine indices of axis-direction neighbours respecting periodic boundary conditions (wrap around) */
            x_w = params.lookup_x_w[jj];
            x_e = params.lookup_x_e[jj];

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
#ifdef BOLTZMANN_ACCURATE
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
#else
                /* directional velocity components */
                u[1] =  u_x;       /* east */
                u[2] =  u_y;       /* north */
                u[5] =  u_x + u_y; /* north-east */
                u[6] = -u_x + u_y; /* north-west */
                /* velocity squared over twice the speed of sound */
                u_sq = (u_x * u_x + u_y * u_y) * 1.5;
                /* equilibrium densities */
                /* zero velocity density: weight w0 */
                d_equ[0] = w0 * local_density * (1.0 - u_sq);

                /* axis speeds: weight w1 */
                /* Note: reusing the otherwise unused u[0] position to pre-compute reused values and speed up the function */
                u[0] = 1.0 + (u[1] * u[1] * 4.5) - u_sq;
                d_equ[1] = w1 * local_density * (u[0] + u[1] * 3.0);
                d_equ[3] = w1 * local_density * (u[0] - u[1] * 3.0);
                u[0] = 1.0 + (u[2] * u[2] * 4.5) - u_sq;
                d_equ[2] = w1 * local_density * (u[0] + u[2] * 3.0);
                d_equ[4] = w1 * local_density * (u[0] - u[2] * 3.0);
                /* diagonal speeds: weight w2 */
                u[0] = 1.0 + (u[5] * u[5] * 4.5) - u_sq;
                d_equ[5] = w2 * local_density * (u[0] + u[5] * 3.0);
                d_equ[7] = w2 * local_density * (u[0] - u[5] * 3.0);
                u[0] = 1.0 + (u[6] * u[6] * 4.5) - u_sq;
                d_equ[6] = w2 * local_density * (u[0] + u[6] * 3.0);
                d_equ[8] = w2 * local_density * (u[0] - u[6] * 3.0);

                /* relaxation step */
                for (kk = 0; kk < NSPEEDS; kk++) {
                    speeds[kk] += params.omega * (d_equ[kk] - speeds[kk]);
                }
                memcpy(dst_speeds, speeds, sizeof(t_speed));
#endif
            }
        }
    }

    return EXIT_SUCCESS;
}

int initialise(t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr, unsigned char** obstacles_ptr, double** av_vels_ptr) {
    FILE *fp;          /* file pointer */
    int ii, jj;        /* generic counters */
    int xx, yy;        /* generic array indices */
    int blocked;       /* indicates whether a cell is blocked by an obstacle */
    int retval;        /* to hold return value for checking */
    double w0, w1, w2; /* weighting factors */

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

    params->lookup_buffer = (int *) malloc(sizeof(int) * (params->ny * 2 + params->nx * 2));
    if (params->lookup_buffer == NULL)
    	die("cannot allocate memory for offset buffer", __LINE__, __FILE__);

    params->lookup_y_s = &params->lookup_buffer[0];
    params->lookup_y_n = &params->lookup_y_s[params->ny];
    params->lookup_x_w = &params->lookup_y_n[params->ny];
    params->lookup_x_e = &params->lookup_x_w[params->nx];

    /* Pre-calculate offsets to speed up memory access in the collision loop */
    for (ii = 0; ii < params->ny; ii++) {
    	params->lookup_y_s[ii] = (ii + 1) % params->ny;
    	params->lookup_y_n[ii] = (ii == 0) ? (ii + params->ny - 1) : (ii - 1);
		for (jj = 0; jj < params->nx; jj++) {
			params->lookup_x_w[jj] = (jj + 1) % params->nx;
			params->lookup_x_e[jj] = (jj == 0) ? (jj + params->nx - 1) : (jj - 1);
		}
	}

    /* main grid */
    *cells_ptr = (t_speed*) malloc(sizeof(t_speed) * (params->ny * params->nx));
    if (*cells_ptr == NULL)
        die("cannot allocate memory for cells", __LINE__, __FILE__);

    /* 'helper' grid, used as scratch space */
    *tmp_cells_ptr = (t_speed*) malloc(sizeof(t_speed) * (params->ny * params->nx));
    if (*tmp_cells_ptr == NULL)
        die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

    /* the map of obstacles */
    *obstacles_ptr = (unsigned char *) malloc(sizeof(unsigned char) * (params->ny * params->nx));
    if (*obstacles_ptr == NULL)
        die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

    /* initialise densities */
    w0 = params->density * 4.0 / 9.0;
    w1 = params->density / 9.0;
    w2 = params->density / 36.0;

    for (ii = 0; ii < params->ny; ii++) {
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

    /* first set all cells in obstacle array to zero */
    for (ii = 0; ii < params->ny; ii++) {
        for (jj = 0; jj < params->nx; jj++) {
            (*obstacles_ptr)[ii * params->nx + jj] = 0;
        }
    }

    /* open the obstacle data file */
    fp = fopen(OBSTACLEFILE, "r");
    if (fp == NULL) {
        die("could not open file obstacles", __LINE__, __FILE__);
    }

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
    }

    /* and close the file */
    fclose(fp);

    /* allocate space to hold a record of the avarage velocities computed at each timestep */
    *av_vels_ptr = (double*) malloc(sizeof(double) * params->maxIters);

    return EXIT_SUCCESS;
}

int finalise(t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr, unsigned char** obstacles_ptr, double** av_vels_ptr) {
    /*
     ** free up allocated memory
     */
    free(params->lookup_buffer);
    params->lookup_buffer = NULL;
    params->lookup_y_s = NULL;
    params->lookup_y_n = NULL;
    params->lookup_x_w = NULL;
    params->lookup_x_e = NULL;

    free(*cells_ptr);
    *cells_ptr = NULL;

    free(*tmp_cells_ptr);
    *tmp_cells_ptr = NULL;

    free(*obstacles_ptr);
    *obstacles_ptr = NULL;

    free(*av_vels_ptr);
    *av_vels_ptr = NULL;

    return EXIT_SUCCESS;
}

double av_velocity(const t_param params, t_speed* cells, unsigned char* obstacles) {
    int ii, jj, offset;     /* generic counters */
    int tot_cells = 0;      /* no. of cells used in calculation */
    double local_density;   /* total density in cell */
    double tot_u_x;         /* accumulated x-components of velocity */
    double *speeds;         /* directional speeds */

    /* initialise */
    tot_u_x = 0.0;

    /* loop over all non-blocked cells */
    for (ii = 0; ii < params.ny; ii++) {
        for (jj = 0; jj < params.nx; jj++) {
            offset = ii * params.nx + jj;
            /* ignore occupied cells */
            if (!obstacles[offset]) {
                speeds = cells[offset].speeds;
                /* local density total */
                local_density = speeds[0] + speeds[1] + speeds[2] + speeds[3] + speeds[4] + speeds[5] + speeds[6] + speeds[7] + speeds[8];
                /* x-component of velocity */
                tot_u_x += (speeds[1] + speeds[5] + speeds[8] - (speeds[3] + speeds[6] + speeds[7])) / local_density;
                /* increase counter of inspected cells */
                ++tot_cells;
            }
        }
    }

    return tot_u_x / (double) tot_cells;
}

double calc_reynolds(const t_param params, t_speed* cells, unsigned char* obstacles) {
    const double viscosity = 1.0 / 6.0 * (2.0 / params.omega - 1.0);
    return av_velocity(params, cells, obstacles) * params.reynolds_dim / viscosity;
}

double total_density(const t_param params, t_speed* cells) {
    int ii, jj, kk;     /* generic counters */
    double total = 0.0; /* accumulator */
    for (ii = 0; ii < params.ny; ii++) {
        for (jj = 0; jj < params.nx; jj++) {
            for (kk = 0; kk < NSPEEDS; kk++) {
                total += cells[ii * params.nx + jj].speeds[kk];
            }
        }
    }

    return total;
}

int write_values(const t_param params, t_speed* cells, unsigned char* obstacles, double* av_vels) {
    FILE* fp; /* file pointer */
    int ii, jj, offset; /* generic counters */
    const double c_sq = 1.0 / 3.0; /* sq. of speed of sound */
    double local_density; /* per grid cell sum of densities */
    double pressure; /* fluid pressure in grid cell */
    double u_x; /* x-component of velocity in grid cell */
    double u_y; /* y-component of velocity in grid cell */
    double * speeds;

  fp = fopen(finalStateFile,"w");
    if (fp == NULL) {
        die("could not open file output file", __LINE__, __FILE__);
    }

    for (ii = 0; ii < params.ny; ii++) {
        for (jj = 0; jj < params.nx; jj++) {
            offset = ii * params.nx + jj;
            /* an occupied cell */
            if (obstacles[offset]) {
                u_x = u_y = 0.0;
                pressure = params.density * c_sq;
            } else /* no obstacle */{
                speeds = cells[offset].speeds;
                local_density = speeds[0] + speeds[1] + speeds[2] + speeds[3] + speeds[4] + speeds[5] + speeds[6] + speeds[7] + speeds[8];
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

    fclose(fp);

  fp = fopen(avVelocityFile,"w");
    if (fp == NULL) {
        die("could not open file output file", __LINE__, __FILE__);
    }
    for (ii = 0; ii < params.maxIters; ii++) {
        fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
    }

    fclose(fp);

    return EXIT_SUCCESS;
}

void die(const char* message, const int line, const char *file) {
    fprintf(stderr, "Error at line %d of file %s:\n", line, file);
    fprintf(stderr, "%s\n", message);
    fflush(stderr);
    exit(EXIT_FAILURE);
}
