#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "common.h"
#include "omp.h"
#include <vector>


//Function that returns the bin index, given the number of bins, and the coordinates of the particle

int find_bin_from_particle(double x_coordinate, double y_coordinate, double size_of_bin, double grid_size_unidirection) 
{
    int x_bin;  
    //figuring out the x coordinate
    x_bin = int(x_coordinate / size_of_bin);   
    int y_bin;
    //figuring out the y coordinate
    y_bin = int(y_coordinate / size_of_bin);
    return y_bin * (grid_size_unidirection/size_of_bin) + x_bin;
}

//
//  benchmarking program
//
int main( int argc, char **argv )
{   

    int navg,nabsavg=0,numthreads; 
    double dmin, absmin=1.0,davg,absavg=0.0;
    
    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        printf( "-s <filename> to specify a summary file name\n" ); 
        printf( "-no turns off all correctness checks and particle output\n");   
        return 0;
    }

    int n = read_int( argc, argv, "-n", 1000 );
    char *savename = read_string( argc, argv, "-o", NULL );
    char *sumname = read_string( argc, argv, "-s", NULL );

    FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
    FILE *fsum = sumname ? fopen ( sumname, "a" ) : NULL;      

    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
    set_size( n );
    init_particles( n, particles );

    //Initialize the empty bins with neighbours
    double size_of_grid = sqrt(n * 0.0005);
    
    //Simple initial guess
    double bin_width = sqrt( 3 * 0.0005);
    int number_of_bins =  int(size_of_grid/bin_width);
     
    //Final estimate of the bin width give 3 - 4 particles per bin
    bin_width = size_of_grid / number_of_bins;
    
    //Now create a list of empty bins, and populate their nearest N.
    int total_number_of_bins = number_of_bins * number_of_bins;
    
    bin_t bins[total_number_of_bins];
    omp_lock_t locks[total_number_of_bins];


    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer( );
    #pragma omp for
    for( int i = 0; i < total_number_of_bins; i++ )
    {

        bin_t bin;
        omp_init_lock(locks+i);   
        //checking for corner cases
        int bin_x = i % number_of_bins;
        int bin_y = int(i/number_of_bins);

        if (bin_x == 0 || bin_x == (number_of_bins-1) || bin_y == 0 || bin_y == (number_of_bins-1) )
        {  
            //Checking the neighbours to see which one violated the boundary
            for (int x = -1; x <= 1; x++) {
                for (int y = -1; y <=1; y++) {
                    if (bin_x+x >= 0 && bin_y+y >= 0 && bin_x+x < number_of_bins && bin_y+y < number_of_bins) {
                        bin.neighbours.push_back(i+x+number_of_bins*y);
                    }
                }
            }   
        } else {
            for (int x = -1; x <= 1; x++) {
                for (int y = -1; y <=1; y++) {
                    bin.neighbours.push_back(i+x+number_of_bins*y);
                }
            }   
        }
        bins[i] = bin;
    }
    #pragma omp parallel private(dmin) 
    {
    numthreads = omp_get_num_threads();
    for( int step = 0; step < NSTEPS; step++ )
        {
           navg = 0;
           davg = 0.0;
           dmin = 1.0;
           #pragma omp for
            for (int i = 0; i < total_number_of_bins; i++) {
                bins[i].particles.clear();

            }
            #pragma omp for 
            for( int i = 0; i < n; i++ ) // Reassigning particles to new bins
                {

                    int index_of_bin =  find_bin_from_particle(particles[i].x,particles[i].y, bin_width,size_of_grid);
            particles[i].ax = particles[i].ay = 0;
                    omp_set_lock(locks+index_of_bin);
                    bins[index_of_bin].particles.push_back(i);
                    omp_unset_lock(locks+index_of_bin);
                }

            //apply the force
            #pragma omp for reduction (+:navg) reduction(+:davg) // Tried schedule(dynamic) but saw no change
            for(int i= 0; i<total_number_of_bins;i++)
            {
                for(std::vector<int>::size_type j = 0; j != bins[i].neighbours.size(); j++) {    
                    for(std::vector<int>::size_type k = 0; k < bins[i].particles.size(); k++) {
                        for(std::vector<int>::size_type f = 0; f != bins[bins[i].neighbours[j]].particles.size(); f++) {
                            apply_force( particles[k], particles[f],&dmin,&davg,&navg);
                        }
                    }     
                }
            }


            //
            //  move particles
            //
            #pragma omp for
            for( int i = 0; i < n; i++ ) 
                move( particles[i] );       
            if( find_option( argc, argv, "-no" ) == -1 )
            {
              //
              // Computing statistical data
              //
              #pragma omp master
              if (navg) {
                absavg +=  davg/navg;
                nabsavg++;
              }
              #pragma omp critical
              if (dmin < absmin) absmin = dmin;
            
              //
              //  save if necessary
              //
              #pragma omp master
              if( fsave && (step%SAVEFREQ) == 0 )
                  save( fsave, n, particles );
            }
        }
    }

    simulation_time = read_timer( ) - simulation_time;
    
    printf( "n = %d,threads = %d, simulation time = %g seconds", n,numthreads, simulation_time);

    if( find_option( argc, argv, "-no" ) == -1 )
    {
      if (nabsavg) absavg /= nabsavg;
    // 
    //  -The minimum distance absmin between 2 particles during the run of the simulation
    //  -A Correct simulation will have particles stay at greater than 0.4 (of cutoff) with typical values between .7-.8
    //  -A simulation where particles don't interact correctly will be less than 0.4 (of cutoff) with typical values between .01-.05
    //
    //  -The average distance absavg is ~.95 when most particles are interacting correctly and ~.66 when no particles are interacting
    //
    printf( ", absmin = %lf, absavg = %lf", absmin, absavg);
    if (absmin < 0.4) printf ("\nThe minimum distance is below 0.4 meaning that some particle is not interacting");
    if (absavg < 0.8) printf ("\nThe average distance is below 0.8 meaning that most particles are not interacting");
    }
    printf("\n");
    
    //
    // Printing summary data
    //
    if( fsum)
        fprintf(fsum,"%d %d %g\n",n,numthreads,simulation_time);

    //
    // Clearing space
    //
    if( fsum )
        fclose( fsum );

    free( particles );
    if( fsave )
        fclose( fsave );
    
    return 0;
}
