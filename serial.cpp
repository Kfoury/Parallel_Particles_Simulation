#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "common.h"
#include <vector>



//Function that returns the bin index, given the number of bins, and the coordinates of the particle

int find_bin_from_particle(double x_coordinate, double y_coordinate, double size_of_bin, double grid_size_unidirection  )
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
    int navg,nabsavg=0;
    double davg,dmin, absmin=1.0, absavg=0.0;

    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
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
    // int number_of_bins = 16;    
    //Final estimate of the bin width give 3 - 4 particles per bin
    bin_width = size_of_grid / number_of_bins;
    
    //Variables to use later: bin_width, size_of_grid, number_of_bins
    
    //Now create a list of empty bins, and populate their nearest N.
    
    int total_number_of_bins = number_of_bins * number_of_bins;
    
    // bin_t *bins = (bin_t* )malloc(total_number_of_bins * sizeof(bin_t));
    bin_t bins[total_number_of_bins];
    // bin_t bin;
    double simulation_time = read_timer( );
    for( int i = 0; i < total_number_of_bins; i++ )
        {
            bin_t bin;   
            //checking for corner cases
            int bin_x = i % number_of_bins;
            int bin_y = int(i/number_of_bins);

            if (bin_x == 0 || bin_x == (number_of_bins-1) || bin_y == 0 || bin_y == (number_of_bins-1) )
            {  
                //Checking the neighbours to see which one violated the boundary
                for (int x = -1; x <= 1; x++) {
                    for (int y = -1; y <=1; y++) {
                        if (bin_x+x >= 0 && bin_y+y >= 0 && bin_x+x < number_of_bins && bin_y+y < number_of_bins) {
                            //bins[i].neighbours.push_back(i+x + number_of_bins*y);
                            bin.neighbours.push_back(i+x+number_of_bins*y);
                        }
                    }
                }   
            } else {
                for (int x = -1; x <= 1; x++) {
                    for (int y = -1; y <=1; y++) {
                        // bins[i].neighbours.push_back(i+x + number_of_bins*y);
                        bin.neighbours.push_back(i+x+number_of_bins*y);
                    }
                }   
            }
            bins[i] = bin;
        }

    //
    //  simulate a number of time steps
    //
    for( int step = 0; step < NSTEPS; step++ )
    {
	   navg = 0;
       davg = 0.0;
	   dmin = 1.0;
        for (int i = 0; i < total_number_of_bins; i++) {
            bins[i].particles.clear();
        }
        for( int i = 0; i < n; i++ )
            {
                int index_of_bin =  find_bin_from_particle(particles[i].x,particles[i].y, bin_width,size_of_grid);
                particles[i].ax = particles[i].ay = 0;
                bins[index_of_bin].particles.push_back(i);
            }

        //apply the force
        for(int i= 0; i<total_number_of_bins;i++)
        {
            // printf("i is   %d\n", i);
            for(std::vector<int>::size_type j = 0; j != bins[i].neighbours.size(); j++) {    
                
                for(std::vector<int>::size_type k = 0; k != bins[i].particles.size(); k++) {
                    for(std::vector<int>::size_type f = 0; f != bins[bins[i].neighbours[j]].particles.size(); f++) {
                        
                        apply_force( particles[k], particles[f],&dmin,&davg,&navg);
                    }   
                }     
            }
        }


        //
        //  move particles
        //
        for( int i = 0; i < n; i++ ) 
            move( particles[i] );		
        if( find_option( argc, argv, "-no" ) == -1 )
        {
          //
          // Computing statistical data
          //
          if (navg) {
            absavg +=  davg/navg;
            nabsavg++;
          }
          if (dmin < absmin) absmin = dmin;
		
          //
          //  save if necessary
          //
          if( fsave && (step%SAVEFREQ) == 0 )
              save( fsave, n, particles );
        }
    }
    simulation_time = read_timer( ) - simulation_time;
    
    printf( "n = %d, simulation time = %g seconds", n, simulation_time);

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
        fprintf(fsum,"%d %g\n",n,simulation_time);
 
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
