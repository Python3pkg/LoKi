#! /usr/bin/env python

import numpy as np
import sys, time
from astropy.table import Table
import loki
import loki_constants 

def prob(sub1 = None, sub2 = None, RAs1 = None, DECs1 = None, RAs2 = None, DECs2 = None,
         dists1 = None, dists2 = None, disterrs1 = None, disterrs2 = None,
         PMras1 = None, PMdecs1 = None, PMras2 = None, PMdecs2 = None,
         PMraerrs1 = None, PMdecerrs1 = None, PMraerrs2 = None, PMdecerrs2 = None,
         RVs1 = None, RVs2 = None, RVerrs1 = None, RVerrs2 = None,
         PM = False, RV = False, DIST = False, DISTERR = False, size = 10000,
         infile = None, outfile = None, nstepsMC = 1000, subset = False, random = False):

    """
    This program reads in (or an input can be used) a list of binaries
    and calculates the number of stars expected in the LOS and volume
    occupied by the binary system by creating "fake" galaxies and
    randomly distributing the stars in that galaxy. It uses the
    rejection method (Press et al. 1992) to distribute the stars using
    the stellar density profile as calculated by Bochanski et al. (2009)
    using SDSS low-mass stars.

    Written by : Saurav Dhital
    Written on : April 17, 2009
    Ported to Python: Chris Theissen, June 29, 2015
    """

    # Start the program and time it
    print('Start Time: ', time.strftime("%I:%M:%S"))
    t_start = time.time()

    # Check if there is kinematic and positional information
    if PMras1 is not None and PM   == False: 
        PM   = True          # Proper motions available

    if RVs1   is not None and RV   == False: 
        RV   = True          # Radial velocities available

    if dists1 is not None and DIST == False: 
        DIST = True          # Distances available

    # Get the file and/or define the parameters
    if infile is None:   # Case where there is no file  

        if RAs1 is None and RAs2 is None:
            raise Exception('Either a file or a list of RADECs must be included')

        else:

            # Check if (numpy) arrays
            if isinstance( RAs1, np.ndarray ) == False:
                RAs1  = np.array( RAs1 ).flatten()
                DECs1 = np.array( DECs1 ).flatten()
                RAs2  = np.array( RAs2 ).flatten()
                DECs2 = np.array( DECs2 ).flatten()
                n     = len( RAs1 )                    # See how long the file is

            if DIST and isinstance( dists1, np.ndarray ) == False:
                dists1  = np.array( dists1 ).flatten()
                dists2  = np.array( dists2 ).flatten()
                if disterrs1 is not None:
                    disterrs1  = np.array( disterrs1 ).flatten()
                    disterrs2  = np.array( disterrs2 ).flatten()

            if PM and isinstance( PMras1, np.ndarray ) == False:
                PMras1     = np.array( PMras1 ).flatten()
                PMdecs1    = np.array( PMdecs1 ).flatten()
                PMras2     = np.array( PMras2 ).flatten()
                PMdecs2    = np.array( PMdecs2 ).flatten()
                PMraerrs1  = np.array( PMraerrs1 ).flatten()
                PMdecerrs1 = np.array( PMdecerrs1 ).flatten()
                PMraerrs2  = np.array( PMraerrs2 ).flatten()
                PMdecerrs2 = np.array( PMdecerrs2 ).flatten()

            if RV and isinstance(RVs1, np.ndarray) == False:
                RVs1 = np.array( RVs1 ).flatten()
                RVs2 = np.array( RVs2 ).flatten()
                RVerrs1 = np.array( RVerrs1 ).flatten()
                RVerrs2 = np.array( RVerrs2 ).flatten()

    else: # Case where there is a file

        binaries = Table.read(infile)    # Read in the file
        # Define the parameters
        RAs1  = binaries['RA1']
        DECs1 = binaries['DEC1']
        RAs2  = binaries['RA2']
        DECs2 = binaries['DEC2']

        if DIST:

            dists1 = binaries['DIST1']
            dists2 = binaries['DIST2']

            if DISTERR == False:

                # Just apply a 20% uncertainty
                disterrs1 = .2*binaries['DIST1']
                disterrs2 = .2*binaries['DIST2']

            else:

                disterrs1 = binaries['DISTERR1']
                disterrs2 = binaries['DISTERR2']

        if PM:
            PMras1     = binaries['PMRA1']
            PMdecs1    = binaries['PMDEC1']
            PMras2     = binaries['PMRA1']
            PMdecs2    = binaries['PMDEC1']
            PMraerrs1  = binaries['PMRAERR1']
            PMdecerrs1 = binaries['PMDECERR1']
            PMraerrs2  = binaries['PMRAERR2']
            PMdecerrs2 = binaries['PMDECERR2']

        if RV:
            RVs1    = binaries['RV1']
            RVs2    = binaries['RV2']
            RVerrs1 = binaries['RVERR1']
            RVerrs2 = binaries['RVERR2']
    
    n = len(RAs1)                                 # See how many candidates there are
    binaries_theta = loki.angdist(RAs1, DECs1, RAs2, DECs2) # Get the angular distances between the pairs

    # **************************************************************************

    print('')
    print('No. of candidate pairs: %s'%n)
    print('No. of MC steps       : %s'%nstepsMC)
    print('')

    # storage arrays
    nstars       = np.zeros(n)                            # stores no. of stars in each LOS
    count_star   = np.zeros( (n, 5), dtype=np.float64)    # stores no. of companions for each LOS
    count_binary = np.zeros( (n, 5), dtype=np.float64)    # stores no. of companions for each LOS

    print('We are at (%s):'%('%'))

    count = 0

    for i in range(0, n):         # loop for each LOS (binary)
    
        ra0    = RAs1[i]       # system properties are subscripted with 0
        dec0   = DECs1[i] 
        theta0 = binaries_theta[i]

        if DIST:  # Check if distances are within uncertainties

            dist0  = 0.5 * (dists1[i] + dists2[i])

            if DISTERR == False:
                # Fudge factor for things without distance uncertainties
                sig_ddist0 = 0.1383 * np.sqrt( dists1[i] ** 2 + dists2[i] ** 2 )

            else:
                sig_ddist0 = np.sqrt( disterrs1[i] ** 2 + disterrs2[i] ** 2 )

        # Get the kinematic information if available
        if RV and PM: # Case with proper motions and radial velocities

            vel0     = 0.5 * np.array( [ PMras1[i]  + PMras2[i],
                                         PMdecs1[i] + PMdecs2[i],
                                         RVs1[i]    + RVs2[i] ] )
            sig_vel0 = np.sqrt( np.array( [ PMraerrs1[i] ** 2   + PMraerrs2[i] ** 2,
                                            PMdecerrs1[i] ** 2  + PMdecerrs2[i] ** 2,
                                            RVerrs1[i] ** 2     + RVerrs2[i] ** 2 ] ) )
            #vel0       = 0.5 * np.array([bry['PMRA'][i][0]  + bry['PMRA'][i][1],
            #                             bry['PMDEC'][i][0] + bry['PMDEC'][i][1],
            #                             bry['RV'][i][0]    + bry['RV'][i][1]])
            #sig_vel0   = np.sqrt( np.array([bry['PMRAERR'][i][0]**2  + bry['PMRAERR'][i][1]**2,
            #                                bry['PMDECERR'][i][0]**2 + bry['PMDECERR'][i][1]**2,
            #                                bry['RVERR'][i][0]**2    + bry['RVERR'][i][1]**2]) )

        if RV == False and PM:  # Case with proper motions but no radial velocities

            vel0     = 0.5 * np.array( [ PMras1[i]  + PMras2[i],
                                         PMdecs1[i] + PMdecs2[i] ] )
            sig_vel0 = np.sqrt( np.array( [ PMraerrs1[i] ** 2   + PMraerrs2[i] ** 2,
                                            PMdecerrs1[i] ** 2  + PMdecerrs2[i] ** 2 ] ) )


        # **********************************************************
        # ********************      CALC PROB    *******************
        # **********************************************************
        # storage arrays
        count_MC  = np.zeros(5, dtype = np.float64)        # store data for each niter
        count_MC2 = np.zeros(5, dtype = np.float64)        # store data for each niter

        # count the number of stars in each cell of length cell_size
        nstars_tot, nstars2, dists = loki.count_nstars(ra0, dec0, full = True) 
        #nstars_tot = np.around(nstars_tot)
        #nstars[i] = nstars_tot

        # Need to create a probability distribution based on the float
        randstar    = np.random.random_sample( size )
        nstars_tot2 = np.zeros( size, dtype=np.int64 ) + np.floor( nstars_tot )
        nstars_tot2[ np.where( randstar < nstars_tot - np.floor( nstars_tot ) ) ] += 1
        nstars_tot3 = np.random.choice(nstars_tot2, size = 1) # Select a random choice of stars (need an integer)
        nstars[i]   = nstars_tot3 # Add the value to the total stars simulated

        for niter in range(0, nstepsMC): # Loop through the MC (Could probably do this more efficiently)

            # This keeps track of the binary probability
            b1, b2, b3, b4, b5 = 0, 0, 0, 0, 0 
            c1, c2, c3, c4, c5 = 0, 0, 0, 0, 0

            ra, dec, dist  = loki.gen_nstars( ra0, dec0, nstars_tot, nstars2, dists )
            theta          = loki.angdist( ra0, dec0, ra, dec )

            # ************** COUNT FOR MATCHES  *******************
            ind1 = np.where( ( theta >= 0 ) & ( theta <= theta0 ) )   # counts all stars within given theta and all d
            c1   = len( ind1[0] )

            if c1 >= 1: 
                b1 += 1

            if DIST:
                ddist = abs( dist0 - dist ) 

                ind2 = np.where( ( theta >= 0) & ( theta <= theta0 ) & 
                                 ( ddist <= sig_ddist0 ) & ( ddist <= 100 ) ) # counts stars within given theta and d
                c2   = len( ind2[0] )
            
                if c2 >= 1: 
                    b2 += 1

            # if kinematics are available                      # NEED TO COME BACK AND FIX THIS
            if c2 > 0 and ( PM or RV ):
    
                R0, T0, Z0 = loki.conv_to_galactic( ra0, dec0, dist0 )         # Convert RADECs to Galactic coordinates
                vel1       = loki.gen_pm( R0, T0, Z0, ra0, dec0, dist0, c2 )   # returns [[pmra], [pmdec],[rv]]

                if PM and RV == False: 
                    vel = np.array( [ vel1[0], vel1[1] ] ) 

                # replicate vel0 to match the dimensions of generated velocity array
                # allows for vector arithmetic
                vel0_arr     = np.tile( vel0, ( c2, 1 ) ).transpose()
                sig_vel0_arr = np.tile( sig_vel0, ( c2, 1 ) ).transpose()

                # difference in binary and simulated velocity in units of sigma
                dVel = abs( vel - vel0_arr ) / sig_vel0_arr

                if PM: # PM match

                    ind3 = np.where( np.sqrt( dVel[0] ** 2 + dVel[1] ** 2 ) <= 2 ) 
                    c3   = len( ind3[0] )

                    if c3 >= 1: 
                        b3 += 1

                else: 
                    c3 = 0
                
                if RV: # RV match

                    ind4 = np.where( dVel[2] <= 1 )
                    c4   = len( ind4[0] )

                    if c4 >= 1: 
                        b4 += 1

                else: 
                    c4 = 0

                if PM and RV: # PM + RV match

                    ind5 = np.where( ( dVel[0] ** 2 + dVel[1] ** 2 <= 2 ) & ( dVel[2] <= 1 ) )
                    c5   = len( ind5[0] )

                    if c5 >= 1: 
                        b5 += 1
                
            else: 
                c3, c4, c5 = 0, 0, 0   # End of one MC step

            # ******************** STORE DATA FOR EACH NITER  ********
            count_MC  += [c1, c2, c3, c4, c5] # This counts the number of stars in the LOS
            count_MC2 += [b1, b2, b3, b4, b5] # This counts the binary (yes or no; there is another star in the LOS)
            count     += 1

            # Keep a running update
            sys.stdout.write("\r%0.4f" %( float(count) / (nstepsMC*n)*100 ) )
            sys.stdout.flush()
            
        # *********************** STORE DATA FOR EACH STAR ***********
        count_star[i,:]   = count_MC
        count_binary[i,:] = count_MC2

    # Convert to probabilities
    prob  = count_star / nstepsMC
    prob2 = count_binary / nstepsMC

    if outfile is None: 
        outfile='Probabilities.csv'

    print('            *************               ')
    print('            *************               ')
    
    # Create the table for results
    if DIST:
        Table1 = Table([RAs1, DECs1, 0.5 * (dists1 + dists2), binaries_theta,
                        prob[:,0], prob[:,1], prob[:,2], prob[:,3], prob[:,4], 
                        prob2[:,0], prob2[:,1], prob2[:,2], prob2[:,3], prob2[:,4], nstars],
                        names = ['RA','DEC','DIST','THETA','P1','P2','P3','P4','P5',
                                 'PB1','PB2','PB3','PB4','PB5','Nstars'])

    else:
        Table1 = Table([RAs1, DECs1, binaries_theta,
                        prob[:,0], prob[:,1], prob[:,2], prob[:,3], prob[:,4], 
                        prob2[:,0], prob2[:,1], prob2[:,2], prob2[:,3], prob2[:,4], nstars],
                        names = ['RA','DEC','THETA','P1','P2','P3','P4','P5',
                                 'PB1','PB2','PB3','PB4','PB5','Nstars'])

    # Write the table locally (overwrite is on by default, scary)
    Table1.write(outfile)

    print('TOTAL TIME TAKEN   : ', (time.time() - t_start) / 3600.,' hours')
    print('TIME TAKEN PER LOS : ', (time.time() - t_start) / (60*n),' minutes')
    print('END TIME           : ', time.strftime("%I:%M:%S"))



def main():
  SLW_6D(sub1 = None, sub2 = None, RAs1 = None, DECs1 = None, RAs2 = None, DECs2 = None,
           dists1 = None, dists2 = None, disterrs1 = None, disterrs2 = None,
           PMras1 = None, PMdecs1 = None, PMras2 = None, PMdecs2 = None,
           PMraerrs1 = None, PMdecerrs1 = None, PMraerrs2 = None, PMdecerrs2 = None,
           RVs1 = None, RVs2 = None, RVerrs1 = None, RVerrs2 = None,
           PM = False, RV = False, DIST = False, DISTERR = False, size = 10000,
           infile = None, outfile = None, nstepsMC = 1000, subset = False, random = False)

if __name__ == '__main__':
  main()
