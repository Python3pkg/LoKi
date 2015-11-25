#! /usr/bin/env python

import numpy as np
#from astropy import units as u
#from astropy.coordinates import SkyCoord
from slw_constants import Rsun, Tsun, Zsun, rho0, au2pc, cell_size, max_dist

########################

def angdist(ra_1, dec_1, ra_2, dec_2):

    # This function (ra, dec) of two objects in the decimal system
    # (degrees) and returns the angular distance between them in
    # arcseconds.

    ra1  = np.deg2rad(ra_1)
    dec1 = np.deg2rad(dec_1) 
    ra2  = np.deg2rad(ra_2) 
    dec2 = np.deg2rad(dec_2)

    dist = np.sqrt( ( (ra1-ra2)**2 * np.cos(dec1) * np.cos(dec2) ) + ( (dec1-dec2) )**2 )

    return np.rad2deg(3600. * dist)

########################

def calc_uvw(R, theta, Z):

    Rdot = 0.
    Tdot = (226. - 0.013*Z - 1.56e-5*Z**2) / R  # will later  be converted to Tdot(Z) 
    Zdot = 0.                                   # typical values for the MW in km/s

    theta = np.deg2rad(theta)  # convert degrees to radians

    Xdot = Rdot * np.cos(theta) - Tdot * np.sin(theta)
    Ydot = -1 * ( Rdot*np.sin(theta) + Tdot * np.cos(theta) )

    return Xdot, Ydot, Zdot

########################

def calc_sigmavel(Z):

    # U_thin,V_thin,W_thin,U_thick,V_thick,W_thick
    # Values obtained by fitting sigma = coeff * Z^power 
    # data from Bochanski et al. (2006)
    # see ~/sdss/uw/velocity_ellipsoid.pro[.ps] FOR fitting algorithm[fit]
    coeff = np.array([7.085, 3.199, 3.702, 10.383, 1.105, 5.403])
    power = np.array([0.276, 0.354, 0.307,  0.285, 0.625, 0.309])

    # calculate sigma_vel from the empirical power-law fit
    sigmaa = coeff * abs(Z)**power

    return sigmaa

########################

def calc_rho(R, Z):

    H_thin    = 260.
    H_thick   = 900.                   # scale heights in pc
    
    L_thin    = 2500.
    L_thick   = 3500.                  # scale lengths in pc
    
    f_thick   = 0.09
    f_halo    = 0.0025                 # stellar density in the solar neighborhood
    
    f_thin    = 1. - f_thick - f_halo
    r_halo    = 2.77                   # halo density gradient
    q         = 0.64                   # halo flattening parameter

    rho_thin  = rho0 * np.exp(-1 * abs(Z-Zsun) / H_thin)  * np.exp(-1 * (R-Rsun) / L_thin)
    rho_thick = rho0 * np.exp(-1 * abs(Z-Zsun) / H_thick) * np.exp(-1 * (R-Rsun) / L_thick)
    rho_halo  = rho0 * ( Rsun / np.sqrt( R**2 + ( Z / q )**2) )**r_halo

    rho       = f_thin * rho_thin + f_thick * rho_thick + f_halo * rho_halo
    frac      = np.array([f_thin * rho_thin, f_thick * rho_thick, f_halo * rho_halo]) / rho

    return rho, frac

########################

def gen_2Dgaussian(mu, sig1, sig2, f1, f2, num):

    n_acc = 0                # number of stars accepted

    while n_acc < num:
        n_lft = num - n_acc    # no. of needed stars

        X  = np.random.rand(n_lft, 1).flatten()*10*sig2 - 5*sig2 # generate random numbers between (-5,5) sigma of a normalized gaussian
        z1 = (X - mu) / sig1
        z2 = (X - mu) / sig2
        G1 = 1 / (np.sqrt(2*np.pi) * sig1)*np.exp(-z1**2 / 2) # thin disk
        G2 = 1 / (np.sqrt(2*np.pi) * sig2)*np.exp(-z2**2 / 2) # thin disk
 
        Px = f1*G1 + f2*G2
        Fx = 1.2 * np.max(Px)

        rand = np.random.rand(n_lft, 1).flatten()    # Generate random uniform numbers
        ind = np.where(rand < Px/Fx)                 # uniform deviate comp function
 
        count = len(ind[0])

        if count != 0:
            if n_acc == 0: Xarr = X[ind].flatten()
            else: Xarr = np.append(Xarr, X[ind].flatten())
            n_acc += count

    return Xarr

########################

def gal_uvw_pm(U=-9999, V=-9999, W=-9999, ra=-9999, dec=-9999, distance=-9999, plx=-9999, lsr=True):

    
    #lsr_vel = np.array([-8.5, 13.38, 6.49]) # Coskunoglu et al. 2011
    #lsr_vel = np.array([-10, 5.25, 7.17]) # Dehnen & Binney 1998
    lsr_vel = np.array([-11.1, 12.24, 7.25]) # Schonrich, Dehnen & Binney 2010

    # Check if (numpy) arrays
    if isinstance(ra, np.ndarray) == False:
        ra = np.array(ra).flatten()
        dec = np.array(dec).flatten()
        U = np.array(U).flatten()
        V = np.array(V).flatten()
        W = np.array(W).flatten()
        distance = np.array(distance).flatten()
        plx = np.array(plx).flatten()
    if ra.size <= 2:
        ra = np.array([ra]).flatten()
        dec = np.array([dec]).flatten()
        U = np.array([U]).flatten()
        V = np.array([V]).flatten()
        W = np.array([W]).flatten()
        distance = np.array([distance]).flatten()
        plx = np.array([plx]).flatten()

    goodDistance = 0

    if -9999 in ra or -9999 in dec:
        raise Exception('ERROR - The RA, Dec (J2000) position keywords must be supplied (degrees)')
    if -9999 in U or -9999 in V or -9999 in W:
        raise Exception('ERROR - UVW space velocities (km/s) must be supplied for each star')
    if -9999 in distance: 
        bad  = np.where(distance <= 0)
        Nbad = len(bad[0])
        if Nbad > 0:
            raise Exception('ERROR - All distances must be > 0')
    else:
        plx = 1e3 / distance          # Parallax in milli-arcseconds
        goodDistance = 1
    if -9999 in plx and -9999 in distance:
        raise Exception('ERROR - Either a parallax or distance must be specified')
    elif -9999 in plx and goodDistance == 0:
        bad  = np.where(plx <= 0.)
        Nbad = len(bad[0])
        if Nbad > 0:
            raise Exception('ERROR - Parallaxes must be > 0')
    
    # convert to radians
    cosd = np.cos( np.deg2rad(dec) )
    sind = np.sin( np.deg2rad(dec) )
    cosa = np.cos( np.deg2rad(ra) )
    sina = np.sin( np.deg2rad(ra) )

    try:
        Nra = len(ra)
        vrad  = np.empty(Nra)
        pmra  = np.empty(Nra)
        pmdec = np.empty(Nra)
    except:
        Nra = 0
        vrad  = np.empty(1)
        pmra  = np.empty(1)
        pmdec = np.empty(1)

    k = 4.74047                     # Equivalent of 1 A.U/yr in km/s   
    t = np.array( [ [ 0.0548755604,  0.8734370902,  0.4838350155], 
                    [ 0.4941094279, -0.4448296300,  0.7469822445], 
                    [-0.8676661490, -0.1980763734,  0.4559837762] ] )

    for i in range(0, len(vrad)):
        a = np.array( [ [cosa[i]*cosd[i], -sina[i], -cosa[i]*sind[i] ], 
                        [sina[i]*cosd[i],  cosa[i], -sina[i]*sind[i] ], 
                        [sind[i]        ,  0      ,  cosd[i]         ] ] )
        b = np.dot(t, a)

        uvw = np.array([ U[i], V[i], W[i] ])
        if lsr == True: uvw = uvw - lsr_vel

        vec = np.dot( np.transpose(uvw), b)

        vrad[i] = vec[0]       
        pmra[i] = vec[1] * plx[i] / k
        pmdec[i]= vec[2] * plx[i] / k


    try: sz = ra.shape
    except: sz = [0]
    if sz[0] == 0:
        vrad = vrad[0]
        pmra = pmra[0]
        pmdec= pmdec[0] 

    return vrad, pmra, pmdec

########################

def gen_pm(R0, T0, Z0, ra0, dec0, dist0, num):
    
    sigmaa    = calc_sigmavel(Z0)                                                       # calculate the UVW velocity dispersions
                                                                                        # returns [U_thin,V_thin,W_thin,U_thick,V_thick,W_thick]
    rho, frac = calc_rho(R0, Z0)                                                        # calc the frac of thin/thick disk stars 
                                                                                        # returns frac = [f_thin, f_thick, f_halo]
    vel       = np.array(calc_uvw(R0, T0, Z0)) - np.array(calc_uvw(Rsun, Tsun, Zsun))   # convert to cartesian velocities            
                                                                                        # returns [U,V,W]

    # draw from both the thin and thick disks for UVW velocities
    U = gen_2Dgaussian(vel[0], sigmaa[0], sigmaa[3], frac[0], 1-frac[0], num)
    V = gen_2Dgaussian(vel[1], sigmaa[1], sigmaa[4], frac[0], 1-frac[0], num)
    W = gen_2Dgaussian(vel[2], sigmaa[2], sigmaa[5], frac[0], 1-frac[0], num)

    # change UVW to pmra and pmdec
    rv, pmra, pmdec = gal_uvw_pm(U = U, V = V, W = W, ra = np.zeros(num)+ra0,
                                 dec = np.zeros(num)+dec0, distance = np.zeros(num)+dist0 )

    return pmra, pmdec, rv

########################

def conv_to_galactic(ra, dec, d):

    r2d = 180/np.pi # radians to degrees

    # Check if (numpy) arrays
    if isinstance(ra, np.ndarray)  == False: ra  = np.array(ra).flatten()
    if isinstance(dec, np.ndarray) == False: dec = np.array(dec).flatten()
    if isinstance(d, np.ndarray)   == False: d   = np.array(d).flatten()

    # Convert values to Galactic coordinates
    """
    c_icrs = SkyCoord(ra = ra*u.degree, dec = dec*u.degree, frame = 'icrs')  # The SLOW Astropy way
    l, b = c_icrs.galactic.l.radian, c_icrs.galactic.b.radian
    """
    l, b = euler(ra, dec) #### PROBABLY A BETTER WAY TO DO THIS WITH ASTROPY
    l, b = np.deg2rad(l), np.deg2rad(b)
    
    r   = np.sqrt( (d * np.cos( b ) )**2 + Rsun * (Rsun - 2 * d * np.cos( b ) * np.cos( l ) ) )
    t   = np.rad2deg( np.arcsin(d * np.sin( l ) * np.cos( b ) / r) )
    z   = Zsun + d * np.sin( b - np.arctan( Zsun / Rsun) )
    
    return r, t, z

########################

def gen_nstars(ra0, dec0, num):

    # ra0, dec0, num - input parameters
    # ra, dec, dist  - output arrays for the generated stars

    # Check if (numpy) arrays
    if isinstance(ra0, np.ndarray) == False:
        ra0 = np.array(ra0)
        dec0 = np.array(dec0)

    n_acc = 0                # number of stars accepted

    while n_acc < num:
        n_lft = num - n_acc # no. of needed stars

        ra1       = ra0  + (np.random.rand(n_lft,1).flatten() - 0.5) * cell_size
        dec1      = dec0 + (np.random.rand(n_lft,1).flatten() - 0.5) * cell_size
        dist1     = np.random.rand(n_lft,1).flatten() * max_dist

        R, T, Z   = conv_to_galactic(ra1, dec1, dist1)

        rho, frac = calc_rho(R,Z)

        rand1 = np.random.rand(n_lft,1).flatten()

        # accept if random number is less than rho(R,Z)/rho0
        ind = np.where(rand1 < rho / rho0)
        count = len(ind[0])

        if count != 0:
            if n_acc == 0:
                ra   = ra1[ind]
                dec  = dec1[ind]
                dist = dist1[ind]
            else:
                ra = np.append(ra, ra1[ind].flatten())
                dec = np.append(dec, dec1[ind].flatten())
                dist = np.append(dist, dist1[ind].flatten())
            n_acc += count

    return ra, dec, dist

########################

def count_nstars(ra, dec):
    
    ddist  = 5                                      # steps in distance in pc
    n      = max_dist / ddist + 1                   # number of steps to take
    dist   = np.arange(n, dtype=np.float) * ddist   # 0 < d < 2500 in 5 pc steps
    
    # create an array to store rho for each d
    rho    = np.empty(n) 
    nstars = np.empty(n) 
 
    # define fractional positions so that rho can be averaged
    x = np.array([-0.5,  0.0,  0.5,  -0.25,  0.25, -0.5,
                   0.0,  0.5, -0.25,  0.25, -0.5,   0.0,  0.5]) * cell_size
    y = np.array([-0.5, -0.5, -0.5,  -0.25, -0.25,  0.0,
                   0.0,  0.0,  0.25,  0.25,  0.5,   0.5,  0.5]) * cell_size

    # change to galactic coordinates (R,Z) from input convert (ra, dec, dist)
    for k in range(0, len(dist)):

        R, T, Z       = conv_to_galactic( ra+x, dec+y, dist[k])
    
        rhoTemp, frac = calc_rho(R,Z)                               # calculate the stellar density
        rho[k]        = np.mean(rhoTemp)
        vol           = np.pi * (1800. * dist[k] * au2pc)**2 * 5.   # volume at that d = pi * r^2 * h
        nstars[k]     = rho[k] * vol

    nstars_tot = np.sum(nstars)
    
    return nstars_tot

########################

def euler(ai, bi, select=1, fk4=False):
    """
    NAME:
        EULER
    PURPOSE:
        Transform between Galactic, celestial, and ecliptic coordinates.
    EXPLANATION:
        Use the procedure ASTRO to use this routine interactively
   
    CALLING SEQUENCE:
         AO, BO = EULER(AI, BI, [SELECT=1, FK4=False])
   
    INPUTS:
          AI - Input Longitude in DEGREES, scalar or vector.  If only two
                  parameters are supplied, then  AI and BI will be modified to
                  contain the output longitude and latitude.
          BI - Input Latitude in DEGREES
   
    OPTIONAL INPUT:
          SELECT - Integer (1-6) specifying type of coordinate transformation.
   
         SELECT   From          To        |   SELECT      From            To
          1     RA-Dec (2000)  Galactic   |     4       Ecliptic      RA-Dec
          2     Galactic       RA-DEC     |     5       Ecliptic      Galactic
          3     RA-Dec         Ecliptic   |     6       Galactic      Ecliptic
   
         If not supplied as a parameter or keyword, then EULER will prompt for
         the value of SELECT
         Celestial coordinates (RA, Dec) should be given in equinox J2000
         unless the /FK4 keyword is set.
    OUTPUTS:
          AO - Output Longitude in DEGREES
          BO - Output Latitude in DEGREES
   
    INPUT KEYWORD:
          /FK4 - If this keyword is set and non-zero, then input and output
                celestial and ecliptic coordinates should be given in equinox
                B1950.
          /SELECT  - The coordinate conversion integer (1-6) may alternatively be
                 specified as a keyword
    NOTES:
          EULER was changed in December 1998 to use J2000 coordinates as the
          default, ** and may be incompatible with earlier versions***.
    REVISION HISTORY:
          Written W. Landsman,  February 1987
          Adapted from Fortran by Daryl Yentis NRL
          Converted to IDL V5.0   W. Landsman   September 1997
          Made J2000 the default, added /FK4 keyword  W. Landsman December 1998
          Add option to specify SELECT as a keyword W. Landsman March 2003
    """

    #   J2000 coordinate conversions are based on the following constants
    #   (see the Hipparcos explanatory supplement).
    #  eps = 23.4392911111d              Obliquity of the ecliptic
    #  alphaG = 192.85948d               Right Ascension of Galactic North Pole
    #  deltaG = 27.12825d                Declination of Galactic North Pole
    #  lomega = 32.93192d                Galactic longitude of celestial equator
    #  alphaE = 180.02322d               Ecliptic longitude of Galactic North Pole
    #  deltaE = 29.811438523d            Ecliptic latitude of Galactic North Pole
    #  Eomega  = 6.3839743d              Galactic longitude of ecliptic equator

    from numpy import array, sin, cos, pi, deg2rad, rad2deg, arctan2, arcsin, minimum
   
    if fk4:
        equinox = '(B1950)'
        psi    = array([0.57595865315, 4.9261918136,
                        0.00000000000, 0.0000000000,
                        0.11129056012, 4.7005372834])
        stheta = array([0.88781538514, -0.88781538514,
                        0.39788119938, -0.39788119938,
                        0.86766174755, -0.86766174755])
        ctheta = array([0.46019978478, 0.46019978478,
                        0.91743694670, 0.91743694670,
                        0.49715499774, 0.49715499774])
        phi    = array([4.9261918136, 0.57595865315,
                        0.0000000000, 0.00000000000,
                        4.7005372834, 0.11129056012])
    else:
        equinox = '(J2000)'
        psi    = array([0.57477043300, 4.9368292465,
                        0.00000000000, 0.0000000000,
                        0.11142137093, 4.71279419371])
        stheta = array([0.88998808748, -0.88998808748,
                        0.39777715593, -0.39777715593,
                        0.86766622025, -0.86766622025])
        ctheta = array([0.45598377618, 0.45598377618,
                        0.91748206207, 0.91748206207,
                        0.49714719172, 0.49714719172])
        phi    = array([4.9368292465, 0.57477043300,
                        0.0000000000, 0.00000000000,
                        4.71279419371, 0.11142137093])
    if select not in [1,2,3,4,5,6]:
        raise ValueError('Select parameter should be an integer between 1 and 6')
    i = select - 1
    a = deg2rad(ai) - phi[i]
    b = deg2rad(bi)
    sb = sin(b)
    cb = cos(b)
    cbsa = cb * sin(a)
    b = -stheta[i] * cbsa + ctheta[i] * sb
    bo = rad2deg(arcsin(minimum(b, 1.0)))
    del b
    a = arctan2(ctheta[i] * cbsa + stheta[i] * sb, cb * cos(a))
    del cb, cbsa, sb
    ao = rad2deg(((a + psi[i] + 4 * pi ) % (2 * pi) ) )

    return ao, bo

########################
