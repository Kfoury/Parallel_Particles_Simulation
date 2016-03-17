#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>
#include "common.h"


#define NUM_THREADS 256

extern double size;
// __device__ int find_bin_from_particle(double x_coordinate, double y_coordinate, double size_of_bin, double grid_size_unidirection  );
// __device__  std::vector<int> find_valid_neighbors(int bin_id, int num_bins);
// __global__ void create_bins(int n, int* bins);

//
//  benchmarking program
//
__device__ int find_bin_from_particle(double x_coordinate, double y_coordinate, double size_of_bin, double grid_size_unidirection  )
{
    
    int x_bin;  
    //figuring out the x coordinate
    x_bin = int(x_coordinate / size_of_bin);
    
    int y_bin;
    //figuring out the y coordinate
    y_bin = int(y_coordinate / size_of_bin);
    
    
    return y_bin * (grid_size_unidirection/size_of_bin) + x_bin;
    
}

__device__ void find_valid_neighbors(int bin_id, int num_bins, int* neighbors) {
  int i = 0;

  for (int j = 0; j < 9; j++) {
    neighbors[i] = -1;
  }
  int bin_x, bin_y;
  bin_x = bin_id % num_bins;
  bin_y = int(bin_id / num_bins);
  for (int x = -1; x <= 1; x ++) {
    for (int y = -1; y <=1; y++) {
      if ((x != 0 || y != 0) && bin_x+x >= 0 && bin_x+x < num_bins && bin_y+y >= 0 && bin_y+y < num_bins) {
        neighbors[i] = bin_id+x+y*num_bins;
        i++; 
      }
    }
  }
}

__global__ void create_bins(int n, int* bins) {
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  if (tid >= n) return;
  bins[tid] = -1;
}

__global__ void create_particles(int n, int* parts, particle_t* particles, int* bins, double bin_size, int grid_size) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n) return;
  int bin = find_bin_from_particle(particles[tid].x, particles[tid].y, bin_size, grid_size);
  parts[tid] = atomicExch(&bins[bin], tid);
}




__device__ void apply_force_gpu(particle_t &particle, particle_t &neighbor)
{
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if( r2 > cutoff*cutoff )
        return;
    //r2 = fmax( r2, min_r*min_r );
    r2 = (r2 > min_r*min_r) ? r2 : min_r*min_r;
    double r = sqrt( r2 );

    //
    //  very simple short-range repulsive force
    //
    double coef = ( 1 - cutoff / r ) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}
__device__ void apply_force(int tid, particle_t* particles, int* parts, int i){
    particle_t* p = &particles[tid];
    while (i != -1)
    {
        apply_force_gpu(*p, particles[i]);
        i = parts[i];
    }
}

__global__ void compute_forces_gpu(particle_t * particles, int* parts, int* bins, int n, int size_of_bin, int grid_size)
{
  // Get thread (particle) ID
  int num_bins = int(grid_size/size_of_bin);
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid >= n) return;
  int bin_id;
  bin_id = find_bin_from_particle(particles[tid].x, particles[tid].y, size_of_bin, grid_size);
  particles[tid].ax = particles[tid].ay = 0;
  int neighbors[9];
  find_valid_neighbors(bin_id, num_bins, neighbors);
  apply_force(tid, particles, parts, bins[bin_id]);
  for (int k = 0; neighbors[k] != -1; k++) {
    apply_force(tid, particles, parts, bins[neighbors[k]]);
  }
  for(int j = 0 ; j < n ; j++)
    apply_force_gpu(particles[tid], particles[j]);

}

__global__ void move_gpu (particle_t * particles, int n, double size)
{

  // Get thread (particle) ID
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid >= n) return;

  particle_t * p = &particles[tid];
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x  += p->vx * dt;
    p->y  += p->vy * dt;

    //
    //  bounce from walls
    //
    while( p->x < 0 || p->x > size )
    {
        p->x  = p->x < 0 ? -(p->x) : 2*size-p->x;
        p->vx = -(p->vx);
    }
    while( p->y < 0 || p->y > size )
    {
        p->y  = p->y < 0 ? -(p->y) : 2*size-p->y;
        p->vy = -(p->vy);
    }

}



int main( int argc, char **argv )
{    
    // This takes a few seconds to initialize the runtime
    cudaThreadSynchronize(); 

    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        return 0;
    }
    
    int n = read_int( argc, argv, "-n", 1000 );

    char *savename = read_string( argc, argv, "-o", NULL );
    
    FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );

    // GPU particle data structure
    particle_t * d_particles;
    cudaMalloc((void **) &d_particles, n * sizeof(particle_t));

    set_size( n );

    init_particles( n, particles );

    cudaThreadSynchronize();
    double copy_time = read_timer( );

    // Copy the particles to the GPU
    cudaMemcpy(d_particles, particles, n * sizeof(particle_t), cudaMemcpyHostToDevice);

    cudaThreadSynchronize();
    copy_time = read_timer( ) - copy_time;
    double size_of_grid = sqrt(n * 0.0005);
    
    //Simple initial guess
    double bin_width = sqrt( 3 * 0.0005);
    int number_of_bins =  int(size_of_grid/bin_width);
    //int number_of_bins =  n % 4 == 0 ? n/4 : n/4 + 1; 
    // int number_of_bins = 16;    
    //Final estimate of the bin width give 3 - 4 particles per bin
    bin_width = size_of_grid / number_of_bins;
    int total_number_of_bins = number_of_bins * number_of_bins;
    int * parts;
    int * bins;
    int blks = (n + NUM_THREADS - 1) / NUM_THREADS;
    int blk2 = (total_number_of_bins + NUM_THREADS - 1) / NUM_THREADS;
    cudaMalloc((void **) &parts, n * sizeof(int));
    cudaMalloc((void **) &bins, total_number_of_bins * sizeof(int));
    cudaThreadSynchronize();
    create_bins <<< blk2, NUM_THREADS >>> (total_number_of_bins, bins);
    create_particles <<< blks, NUM_THREADS >>> (n, parts, d_particles, bins, bin_width, size_of_grid);

    
    //
    //  simulate a number of time steps
    //
    cudaThreadSynchronize();
    double simulation_time = read_timer( );

    for( int step = 0; step < NSTEPS; step++ )
    {
        //
        //  compute forces
        //

      compute_forces_gpu <<< blks, NUM_THREADS >>> (d_particles, parts, bins, n, bin_width, size_of_grid);
        
        //
        //  move particles
        //
      move_gpu <<< blks, NUM_THREADS >>> (d_particles, n, size);
      
      create_bins <<< blk2, NUM_THREADS >>> (total_number_of_bins, bins);
      create_particles <<< blks, NUM_THREADS >>> (n, parts, d_particles, bins, bin_width, size_of_grid);

        //
        //  save if necessary
        //
        if( fsave && (step%SAVEFREQ) == 0 ) {
      // Copy the particles back to the CPU
            cudaMemcpy(particles, d_particles, n * sizeof(particle_t), cudaMemcpyDeviceToHost);
            save( fsave, n, particles);
        }
    }
    
    cudaThreadSynchronize();
    simulation_time = read_timer( ) - simulation_time;
    
    printf( "CPU-GPU copy time = %g seconds\n", copy_time);
    printf( "n = %d, simulation time = %g seconds\n", n, simulation_time );
    
    free( particles );
    cudaFree(d_particles);
    cudaFree(parts);
    cudaFree(bins);
    if( fsave )
        fclose( fsave );
    
    return 0;
}
