#! /usr/bin/env python

import numpy as np
import sys, time
from astropy.table import Table
#import matplotlib.pyplot as plt
"""
from angdist import angdist
from calc_rho import calc_rho
from calc_sigmavel import calc_sigmavel
from calc_uvw import calc_uvw
from conv_to_galactic import conv_to_galactic
from gen_nstars import gen_nstars
from count_nstars import count_nstars
"""
from slw_routines import angdist, calc_uvw, calc_sigmavel, calc_rho, gen_nstars, conv_to_galactic, count_nstars, gen_pm
from slw_constants import Rsun, Tsun, Zsun, rho0, au2pc, cell_size, max_dist

def SLW_6D(sub1 = None, sub2 = None, RAs1 = None, DECs1 = None, RAs2 = None, DECs2 = None,
           dists1 = None, dists2 = None, disterrs1 = None, disterrs2 = None,
           PMras1 = None, PMdecs1 = None, PMras2 = None, PMdecs2 = None,
           PMraerrs1 = None, PMdecerrs1 = None, PMraerrs2 = None, PMdecerrs2 = None,
           RVs1 = None, RVs2 = None, RVerrs1 = None, RVerrs2 = None,
           PM = False, RV = False, DIST = False, DISTERR = False,
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

    print('######################################################################################################## Starting')

    # **************************************************************************
    # **************************************************************************

    # Start the program and time it
    print('Start Time: ', time.strftime("%I:%M:%S"))
    t_start = time.time()

    # Check if there is kinematic information
    #PM, RV, DIST = 'no', 'no', 'no'
    if PMras1 != None and PM   == False: PM   = True
    if RVs1   != None and RV   == False: RV   = True
    if dists1 != None and DIST == False: DIST = True

    # Get the file and/or define the parameters
    if infile == None:              
        if RAs == None:
            raise Exception('Either a file or a list of RADECs must be included')
        else:
            # Check if (numpy) arrays
            if isinstance(RAs1, np.ndarray) == False:
                RAs1  = np.array(RAs1).flatten()
                DECs1 = np.array(DECs1).flatten()
                RAs2  = np.array(RAs2).flatten()
                DECs2 = np.array(DECs2).flatten()
                n     = len(RAs1)                    # See how long the file is
            if DIST == True and isinstance(dists1, np.ndarray) == False:
                dists1  = np.array(dists1).flatten()
                dists2  = np.array(dists2).flatten()
                if disterrs1 != None:
                    disterrs1  = np.array(disterrs1).flatten()
                    disterrs2  = np.array(disterrs2).flatten()
            if PM == True and isinstance(PMras1, np.ndarray) == False:
                PMras1     = np.array(PMras1).flatten()
                PMdecs1    = np.array(PMdecs1).flatten()
                PMras2     = np.array(PMras2).flatten()
                PMdecs2    = np.array(PMdecs2).flatten()
                PMraerrs1  = np.array(PMraerrs1).flatten()
                PMdecerrs1 = np.array(PMdecerrs1).flatten()
                PMraerrs2  = np.array(PMraerrs2).flatten()
                PMdecerrs2 = np.array(PMdecerrs2).flatten()
            if RV == True and isinstance(RVs1, np.ndarray) == False:
                RVs1 = np.array(RVs1).flatten()
                RVs2 = np.array(RVs2).flatten()
                RVerrs1 = np.array(RVerrs1).flatten()
                RVerrs2 = np.array(RVerrs2).flatten()

    else:
        bry = Table.read(infile)    # Read in the file
        # Define the parameters
        RAs1  = bry['RA1']
        DECs1 = bry['DEC1']
        RAs2  = bry['RA2']
        DECs2 = bry['DEC2']
        if DIST == True:
            dists1 = bry['DIST1']
            dists2 = bry['DIST2']
            if DISTERR == True:
                # Just apply a 20% uncertainty
                disterrs1 = .2*bry['DIST1']
                disterrs2 = .2*bry['DIST2']
                #disterrs1 = bry['DISTERR'][:,0]
                #disterrs2 = bry['DISTERR'][:,1]
        if PM == True:
            PMras1     = bry['PMRA1']
            PMdecs1    = bry['PMDEC1']
            PMras2     = bry['PMRA1']
            PMdecs2    = bry['PMDEC1']
            PMraerrs1  = bry['PMRAERR1']
            PMdecerrs1 = bry['PMDECERR1']
            PMraerrs2  = bry['PMRAERR2']
            PMdecerrs2 = bry['PMDECERR2']
        if RV == True:
            RVs1    = bry['RV'][:,0]
            RVs2    = bry['RV'][:,1]
            RVerrs1 = bry['RVERR'][:,0]
            RVerrs2 = bry['RVERR'][:,1]
    
    n = len(RAs1)                # See how many candidates there are
    # **************************************************************************
    # What is this doing? (Getting a subset if not doing the whole file)
    """
    chunk_size = 10#1000

    x0 = (chunk - 1) * chunk_size
    x1 = x0 + chunk_size - 1
    if chunk == 103: x1 = n-1
    print(chunk, x0, x1)
    bry = bry[x0:x1]
    bry_theta = bry_theta[x0:x1]
    """

    if (random == True and subset == False) or (random == True and full == True): # do the model on a randomized subset (equivalent statements)
        if sub1 == None:
            raise Exception('Must include a subset to randomize (e.g., sub1 = 100)')
        indices = np.around(np.random.rand(sub1, 1).flatten()*n)
        RAs1  = RAs1[indices]
        DECs1 = DECs1[indices]
        RAs2  = RAs2[indices]
        DECs2 = DECs2[indices]
        if DIST == True:
            dists1 = dists1[indices]
            dists2 = dists2[indices]
            if DISTERR == True:
                disterrs1 = disterrs1[indices]
                disterrs2 = disterrs2[indices]
        if PM == True:
            PMras1     = PMras1[indices]
            PMdecs1    = PMdecs1[indices]
            PMras2     = PMras2[indices]
            PMdecs2    = PMdecs2[indices]
            PMraerrs1  = PMraerrs1[indices]
            PMdecerrs1 = PMdecerrs1[indices]
            PMraerrs2  = PMraerrs2[indices]
            PMdecerrs2 = PMdecerrs2[indices]
        if RV == True:
            RVs1    = RVs1[indices]
            RVs2    = RVs2[indices]
            RVerrs1 = RVerrs1[indices]
            RVerrs2 = RVerrs2[indices]
        #bry = bry[np.around(np.random.rand(sub1, 1).flatten()*n)]   
        #bry_theta = bry_theta[np.around(np.random.rand(sub1, 1).flatten()*n)]
    elif random == False and subset == True: # do the model on a subset
        if sub1 == None or sub2 == None:
            raise Exception('Must include a subset to use (e.g., sub1 = 50, sub2 = 99)')
        if sub1 > sub2: subs1, subs2 = sub2, sub1
        else: subs1, subs2 = sub1, sub2
        RAs1  = RAs1[subs1:subs2]
        DECs1 = DECs1[subs1:subs2]
        RAs2  = RAs2[subs1:subs2]
        DECs2 = DECs2[subs1:subs2]
        if DIST == True:
            dists1 = dists1[subs1:subs2]
            dists2 = dists2[subs1:subs2]
            if DISTERR == True:
                disterrs1 = disterrs1[subs1:subs2]
                disterrs2 = disterrs2[subs1:subs2]
        if PM == True:
            PMras1     = PMras1[subs1:subs2]
            PMdecs1    = PMdecs1[subs1:subs2]
            PMras2     = PMras2[subs1:subs2]
            PMdecs2    = PMdecs2[subs1:subs2]
            PMraerrs1  = PMraerrs1[subs1:subs2]
            PMdecerrs1 = PMdecerrs1[subs1:subs2]
            PMraerrs2  = PMraerrs2[subs1:subs2]
            PMdecerrs2 = PMdecerrs2[subs1:subs2]
        if RV == True:
            RVs1    = RVs1[subs1:subs2]
            RVs2    = RVs2[subs1:subs2]
            RVerrs1 = RVerrs1[subs1:subs2]
            RVerrs2 = RVerrs2[subs1:subs2]
        #bry       = bry[subs1:subs2] 
        #bry_theta = bry_theta[subs1:subs2]
        
    n = len(RAs1)   # See how long the file is
    bry_theta = angdist(RAs1, DECs1, RAs2, DECs2)
    #bry_theta = angdist(bry['RA'][:,0], bry['DEC'][:,0], bry['RA'][:,1], bry['DEC'][:,1])
    # **************************************************************************

    print('')
    print('No. of candidate pairs: %s'%n)
    print('No. of MC steps       : %s'%nstepsMC)
    print('')

    # storage arrays
    nstars       = np.zeros(n)                            # stores no. of stars in each 30' x 30'LOS
    count_star   = np.zeros( (n, 5), dtype=np.float64)    # stores no. of companions for each LOS
    count_binary = np.zeros( (n, 5), dtype=np.float64)    # stores no. of companions for each LOS

    print('We are at (%s):'%('%'))
    count = 0
    #n = 1 # TESTING!!!!
    for i in range(0, n):         # loop for each LOS (binary)

        #print('i:', i)
    
        ra0    = RAs1[i]       # system properties are subscripted with 0
        dec0   = DECs1[i] 
        theta0 = bry_theta[i]
        if DIST == True:
            dist0  = 0.5 * (dists1[i] + dists2[i])
            if DISTERR == False:
                sig_ddist0 = 0.1383 * np.sqrt(dists1[i]**2 + dists2[i]**2)
            else:
                sig_ddist0 = np.sqrt(disterrs1[i]**2 + disterrs2[i]**2)
        #dist0      = 0.5 * (bry['DIST'][i][0] + bry['DIST'][i][1])
        #sig_ddist0 = 0.1383 * np.sqrt(bry['DIST'][i][0]**2 + bry['DIST'][i][1]**2)

        # Get the kinematic information if available
        if RV == True and PM == True:
            vel0     = 0.5 * np.array([PMras1[i]  + PMras2[i],
                                       PMdecs1[i] + PMdecs2[i],
                                       RVs1[i]    + RVs2[i]])
            sig_vel0 = np.sqrt( np.array([PMraerrs1[i]**2  + PMraerrs2[i]**2,
                                          PMdecerrs1[i]**2  + PMdecerrs2[i]**2,
                                          RVerrs1[i]**2  + RVerrs2[i]**2]) )
            #vel0       = 0.5 * np.array([bry['PMRA'][i][0]  + bry['PMRA'][i][1],
            #                             bry['PMDEC'][i][0] + bry['PMDEC'][i][1],
            #                             bry['RV'][i][0]    + bry['RV'][i][1]])
            #sig_vel0   = np.sqrt( np.array([bry['PMRAERR'][i][0]**2  + bry['PMRAERR'][i][1]**2,
            #                                bry['PMDECERR'][i][0]**2 + bry['PMDECERR'][i][1]**2,
            #                                bry['RVERR'][i][0]**2    + bry['RVERR'][i][1]**2]) )

        if RV == False and PM == True:
            vel0     = 0.5 * np.array([PMras1[i]  + PMras2[i],
                                       PMdecs1[i] + PMdecs2[i]])
            sig_vel0 = np.sqrt( np.array([PMraerrs1[i]**2  + PMraerrs2[i]**2,
                                          PMdecerrs1[i]**2  + PMdecerrs2[i]**2]) )


        # **********************************************************
        # ********************      CALC PROB    *******************
        # **********************************************************
        # storage arrays
        count_MC  = np.zeros(5, dtype = np.float64)        # store data for each niter
        count_MC2 = np.zeros(5, dtype = np.float64)        # store data for each niter

        # count the number of stars in each cell of length cell_size
        nstars_tot, nstars2, dists = count_nstars(ra0, dec0, cellsize = None, number = True) 
        nstars_tot = np.around(nstars_tot)
        nstars[i] = nstars_tot

        for niter in range(0, nstepsMC):

            # This keeps track of the binary probability
            b1,b2,b3,b4,b5 = 0,0,0,0,0 

            #ra, dec, dist  = gen_nstars(ra0, dec0, nstars[i])     # DO WE WANT TO PARALLELIZE THIS PART? (This part takes the longest)
            ra, dec, dist  = gen_nstars(ra0, dec0, nstars_tot, nstars2, dists, cellsize = None)
            theta          = angdist(ra0, dec0, ra, dec)
            ddist          = abs(dist0 - dist) 
        
            # ************** COUNT FOR MATCHES  *******************
            ind1 = np.where( (theta >= 0) & (theta <= theta0) )       # counts all stars within given theta and all d
            c1   = len(ind1[0])
            if c1 >= 1: b1 += 1
            ind2 = np.where( (theta >= 0) & (theta <= theta0) & 
                             (ddist <= sig_ddist0) & (ddist <= 100) ) # counts stars within given theta and d
            c2   = len(ind2[0])
            if c2 >= 1: b2 += 1

            # if kinematics are available                      # NEED TO COME BACK AND FIX THIS
            if c2 > 0 and (PM == True or RV == True):
    
                R0, T0, Z0 = conv_to_galactic(ra0, dec0, dist0)         # Convert RADECs to Galactic coordinates
                vel1       = gen_pm(R0, T0, Z0, ra0, dec0, dist0, c2)   # returns [[pmra], [pmdec],[rv]]

                if PM == True and RV == False: vel = np.array([vel1[0], vel1[1]]) 

                # replicate vel0 to match the dimensions of generated velocity array
                # allows for vector arithmetic
                vel0_arr     = np.tile(vel0, (c2, 1) ).transpose()
                sig_vel0_arr = np.tile(sig_vel0, (c2, 1) ).transpose()

                # difference in binary and simulated velocity in units of sigma
                dVel = abs(vel - vel0_arr) / sig_vel0_arr

                if PM == True: # PM match
                    ind3 = np.where( np.sqrt(dVel[0]**2 + dVel[1]**2) <= 2) 
                    c3 = len(ind3[0])
                    if c3 >= 1: b3 += 1
                else: c3 = 0
                if RV == True: # RV match
                    ind4 = np.where(dVel[2] <= 1)
                    c4 = len(ind4[0])
                    if c4 >= 1: b4 += 1
                else: c4 = 0
                if PM == True and RV == True: # PM+RV match
                    ind5 = np.where( (dVel[0]**2 + dVel[1]**2 <= 2) & (dVel[2] <= 1) )
                    c5 = len(ind5[0])
                    if c5 >= 1: b5 += 1
                
            else: c3, c4, c5 = 0, 0, 0   # End of one MC step

            # ******************** STORE DATA FOR EACH NITER  ********
            count_MC  += [c1,c2,c3,c4,c5] # This counts the number of stars in the LOS
            count_MC2 += [b1,b2,b3,b4,b5] # This counts the binary (yes or no; there is another star in the LOS)
            count     += 1

            # Keep a running update
            sys.stdout.write("\r%0.4f" %( float(count) / (nstepsMC*n)*100 ) )
            sys.stdout.flush()
            
        # *********************** STORE DATA FOR EACH STAR ***********
        count_star[i,:]   = count_MC
        count_binary[i,:] = count_MC2

        # print update every 5%
        #if i % np.around(0.05 * n) == 0:
            #print((time.time() - t_start) / (60*n), (100.*i/n),'% done. N = ', i+1)

    prob  = count_star / nstepsMC
    prob2 = count_binary / nstepsMC

    chunk=0
    if outfile == None: outfile='slw6D.csv'

    """
    COMMENT = 'RA,DEC         DIST     THETA         P1            P2            P3            P4            P5      Nstars'
    FORMAT = '(2(f13.6),f12.1,f8.2,5(f14.5),i8)'
    # ### RA   --> Right Ascension of simulated ellipsoid
    # ### DEC  --> Declination of simulated ellipsoid
    # ### DIST --> Distance of simulated ellipsoid
    # ### P1 --> P(chance alignment | theta)
    # ### P2 --> P(chance alignment | theta, distance)
    # ### P3 --> P(chance alignment | theta, distance, mu)
    # ### P4 --> P(chance alignment | theta, distance, RV)
    # ### P5 --> P(chance alignment | theta, distance, mu, RV)
    # ### Nstars --> No. of stars simulated

    f = open(outfile, 'a')
    f.write(COMMENT)
    print(' ')
    print('Printing to... ', outfile)
    for i in range(0, n):
        f.write(bry['RA'][i][0], bry['DEC'][i][0], 0.5 * (bry['DIST'][i][0]+bry['DIST'][i][1]), bry_theta[i],
                prob[i,0], prob[i,1], prob[i,2], prob[i,3], prob[i,4], nstars[i])
    f.close()
    """

    print('            *************               ')
    print('            *************               ')
    #forprint,bry.ra[0],bry.dec[0],bry.avg_dist,bry.theta,prob[*,0],prob[*,1],nstars,format=format
    
    Table1 = Table([RAs1, DECs1, 0.5 * (dists1 + dists2), bry_theta,
                    prob[:,0], prob[:,1], prob[:,2], prob[:,3], prob[:,4], 
                    prob2[:,0], prob2[:,1], prob2[:,2], prob2[:,3], prob2[:,4], nstars],
                    names = ['RA','DEC','DIST','THETA','P1','P2','P3','P4','P5',
                             'PB1','PB2','PB3','PB4','PB5','Nstars'])
    Table1.write(outfile)#, overwrite=True)

    print('TOTAL TIME TAKEN   : ', (time.time() - t_start) / 3600.,' hours')
    print('TIME TAKEN PER LOS : ', (time.time() - t_start) / (60*n),' minutes')
    print('END TIME           : ', time.strftime("%I:%M:%S"))



def main():
  SLW_6D(sub1 = None, sub2 = None, RAs1 = None, DECs1 = None, RAs2 = None, DECs2 = None,
           dists1 = None, dists2 = None, disterrs1 = None, disterrs2 = None,
           PMras1 = None, PMdecs1 = None, PMras2 = None, PMdecs2 = None,
           PMraerrs1 = None, PMdecerrs1 = None, PMraerrs2 = None, PMdecerrs2 = None,
           RVs1 = None, RVs2 = None, RVerrs1 = None, RVerrs2 = None,
           PM = False, RV = False, DIST = False, DISTERR = False,
           infile = None, outfile = None, nstepsMC = 1000, subset = False, random = False)

if __name__ == '__main__':
  main()
