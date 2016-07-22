
# -*- coding: utf-8 -*-

"""
LoKi (Low-mass Kinematics): A galactic model for simulating low-mass stellar populations.
Based off the model used for the SLoWPoKES sample (Dhital et al. 2010)
"""

# Import needed packages and functions
from __future__ import print_function, division
import time, os, random
import numpy as np
import scipy.interpolate as interpolate
import loki_constants as constants
import matplotlib.pyplot as plt
from astropy.table import Table


########################################################################


def densities(lffile = None, colors = None, interp = True, size = 10000):

    '''
    Reads in a file with a luminosity function and returns stellar density
    '''

    if lffile is None:
        lffile = os.path.dirname(os.path.realpath(__file__)) + \
                 '/resources/LFs/Single_r_LF_Interp.txt'
        colormin, colormax = 0.5, 4.4 # Approximate color range for the B10 relationship

        if colors is None:
            colors = [colormin, colormax]

    # Check if the file exists
    if os.path.isfile(lffile) is False:
        raise LookupError("File %s does not exist"%lffile)

    """ Not implemented yet, need to get color range from LF
    if colors is None: # Do the full color range
        colors = [colormin, colormax]
    """
    
    if len(colors) != 2: # If colors are defined, make sure they are good
        raise ValueError("colors range ", colors, " should be two values (x1, x2)")

    if min(colors) < colormin or max(colors) > colormax: # check colors are within our range
        raise ValueError("colors range ", colors, " needs to be between %s and %s"%(colormin, colormax))

    # Read in the luminosity function
    t       = Table.read(lffile, format='ascii.csv') 

    # Pull the values from the table
    AbsMag  = t['M_']
    PhiMean = t['Phi_Mean']
    PhiMin  = t['Phi_Min']
    PhiMax  = t['Phi_Max']

    # Interpolate the LF
    if interp is True: # Currently the only version that is implemented
        phiave  = interpolate.InterpolatedUnivariateSpline( AbsMag, PhiMean, k=1)
        philow  = interpolate.InterpolatedUnivariateSpline( AbsMag, PhiMin,  k=1)
        phihigh = interpolate.InterpolatedUnivariateSpline( AbsMag, PhiMax,  k=1)

    # Need to convert colors to absolute magnitudes
    # What is this in absolute (r-band) magnitude?
    # This we know (to ~20%) based off the Mr(r-z) [Bochanski et al. 2010]
    z1     = np.array([- 0.08635, 0.4340, 2.474, 5.190]) 
    M      = np.poly1d(z1)
    M1, M2 =  M(colors[0]), M(colors[1])

    # Integrate the density function
    # First we pull a random phi from a triangular function.
    # 'size' defines how fine you want the interpolation grid to be 
    # (bigger is better, but computationally expensive)
    phi       = np.array( [ np.random.triangular( philow(x), phiave(x), phihigh(x), size ) \
                            for x in np.linspace(M1, M2, size)  ] )
    densities = np.array( [ np.trapz( x = np.linspace(M1, M2, size), y = phi[:,i]) * 1e-3 \
                            for i in range(size) ] )

    return densities


########################################################################


def count_nstars(ra, dec, rho0 = None, cellsize = None, full = False, 
                 maxdist = None, mindist = None, range1 = False):

    '''
    Takes a positions (ra,dec - either ranges or a cellsize), 
    a density, and (optionally) a distance range, and computes 
    the number of stars within the volume.

    This also returns the distribution of distances along the line
    of sight to be used when randomly assigning distances to stars.
    '''

    # If the density isn't set, get it from the constants file
    if rho0 is None: 
        rho0 = constants.rho0
    
    # Get the cell size, or use the default of 30' x 30' 
    if cellsize is None: 
        cellsize = constants.cell_size

    # Check if RADEC are arrays, and if so sort them
    if isinstance(ra, np.ndarray) is False:
        ra = np.array([ra]).flatten()
        ra = np.sort(ra)
    if isinstance(dec, np.ndarray) is False:
        dec = np.array([dec]).flatten()
        dec = np.sort(dec)

    # Check if it is a range (otherwise it should be a cellsize)
    if len(ra) == 2 and len(dec) == 2:
        range1 = True

    # Throw an exception if the ranges are not equal
    if len(ra) == 2 and len(dec) != 2 or \
       len(ra) != 2 and len(dec) == 2:
        raise SyntaxError("RA and DEC ranges must both be of length 2")

    # Throw an error if it there is no range or cellsize
    if range1 is not True and cellsize is None:
        raise SyntaxError("Either a range or a cellsize must be defined")

    # Grab the distance limits from the file if non given
    if mindist is None:
        mindist = constants.min_dist
    if maxdist is None:
        maxdist = constants.max_dist

    dist    = np.arange(mindist, maxdist+constants.ddist, \
                        constants.ddist, dtype=np.float)      # min_dist < d < max_dist in X pc steps
    n       = len(dist)                                       # number of steps to take
    deg2rad = np.pi / 180.                                    # Convert degrees to radians
    
    # create an array to store rho for each distance
    rho    = np.empty(n, dtype=np.float) 
    nstars = np.empty(n, dtype=np.float) 
 
    # Define a grid which will be used to average rho
    # Creart a meshgrid if an RADEC range is used
    if range1 is True:
        nx, ny = (100, 100)
        xv     = np.linspace(ra[0], ra[1], nx)
        yv     = np.linspace(dec[0], dec[1], ny)
        x, y   = np.meshgrid(xv, yv)

    else: # define fractional positions if there is a cellsize
        x = np.array([-0.5,  0.0,  0.5,  -0.25,  0.25, -0.5,
                       0.0,  0.5, -0.25,  0.25, -0.5,   0.0,  0.5]) * cellsize
        y = np.array([-0.5, -0.5, -0.5,  -0.25, -0.25,  0.0,
                       0.0,  0.0,  0.25,  0.25,  0.5,   0.5,  0.5]) * cellsize

    for k in range(0, len(dist)): # Step through each distance to integrate out

        # Convert coordinates to galactic cylindrical
        if range1 is True:
            R, T, Z       = conv_to_galactic( x, y, dist[k] )                 
        else: # The case for the cellsize
            R, T, Z       = conv_to_galactic( ra+x, dec+y, dist[k] )          
        
        rhoTemp, frac     = calc_rho( R, Z, rho0=rho0 )        # Calculate the stellar density
        rho[k]            = np.mean( rhoTemp )                 # Take the mean density across the volume
        
        if range1 is True:
            vol     = 1/3. * ( ( dist[k] + constants.ddist ) ** 3 - dist[k] ** 3 ) * \
                      ( np.cos( ( 90. - dec[0] ) * deg2rad ) - np.cos( ( 90. - dec[1] ) * deg2rad ) ) * \
                      ( ra[0] - ra[1] ) * deg2rad
            if vol  < 0: 
                vol = abs(vol) # This is an absoultely downright dirty fix

        else: # The case for a gridsize
            vol     = 1/3. * ( ( dist[k] + constants.ddist ) ** 3 - dist[k] ** 3) * \
                      ( np.cos( ( 90. - dec - cellsize / 2. ) * deg2rad ) -np.cos( ( 90. - dec + cellsize / 2.) * deg2rad) ) * \
                      ( ( ra + cellsize / 2. ) - ( ra - cellsize / 2. ) ) * deg2rad
            if vol  < 0: 
                vol = abs(vol)
        nstars[k]   = rho[k] * vol     # Add the number of stars (density * volume)

    nstars_tot = np.sum(nstars)        # Compute the total number of stars within the volume
    
    if full is True: # This says you want the distributions of stars and distances
        return nstars_tot, nstars, dist + constants.ddist / 2.
    else: 
        return nstars_tot


########################################################################


class stars(object):

    '''
    Description: Primary class for containing information about the simulated stars.

    Example:
       >>> import loki
       >>> import numpy as np
       >>> ra, dec = 10, 100                                                       # Pick ra,dec values
       >>> densities = loki.densities()                                            # Get back densities
       >>> density = np.random.choice(densities, size = 1)                         # Randomly pick a density 
       >>> n, nums, dists = loki.count_nstars(ra, dec, rho0=density, full = True)  # Compute stellar counts along ra,dec = 10,100
       >>> stars = loki.stars(ra, dec, n, nums, dists)                             # Create stars with given parameters
    '''

    def __init__(self, ra0, dec0, num, nstars, dists):


        ra1, dec1, dist1 = gen_nstars(ra0, dec0, num, nstars, dists)

        self.ra    = ra1
        self.dec   = dec1
        self.dist  = dist1

        l, b = radec2lb(self.ra, self.dec)

        self.l     = l
        self.b     = b

        r, t, z = conv_to_galactic(self.ra, self.dec, self.dist)

        self.R     = r
        self.T     = t
        self.Z     = z

        pmra, pmdec, rv, U, V, W  = gen_pm2(self.R, self.T, self.Z, self.ra, self.dec, self.dist, UVW=True)

        self.pmra  = pmra
        self.pmdec = pmdec
        self.rv    = rv
        self.U     = U
        self.V     = V
        self.W     = W
    

########################################################################


def angdist(ra_1, dec_1, ra_2, dec_2):

    '''
    This function takes the ra,dec of two objects in the degrees
    and returns the angular distance between them in arcseconds.
    '''

    ra1  = np.deg2rad( ra_1 )
    dec1 = np.deg2rad( dec_1 ) 
    ra2  = np.deg2rad( ra_2 )  
    dec2 = np.deg2rad( dec_2 )

    dist = np.sqrt( ( ( ra1 - ra2 ) ** 2 * np.cos( dec1 ) * np.cos( dec2 ) ) + ( ( dec1 - dec2 ) ) ** 2 )

    return np.rad2deg( 3600. * dist )


########################################################################


def conv_to_galactic(ra, dec, d):

    '''
    Function to convert ra, dec, and distances into 
    Galactocentric coordinates R, theta, Z.
    '''

    r2d = 180. / np.pi # radians to degrees

    # Check if (numpy) arrays
    if isinstance(ra, np.ndarray)  == False:
        ra  = np.array(ra).flatten()
    if isinstance(dec, np.ndarray) == False:
        dec = np.array(dec).flatten()
    if isinstance(d, np.ndarray)   == False:
        d   = np.array(d).flatten()

    # Convert values to Galactic coordinates
    """
    # The SLOOOOOOOOOOW Astropy way
    c_icrs = SkyCoord(ra = ra*u.degree, dec = dec*u.degree, frame = 'icrs')  
    l, b = c_icrs.galactic.l.radian, c_icrs.galactic.b.radian
    """
    l, b = radec2lb(ra, dec)
    l, b = np.deg2rad(l), np.deg2rad(b)
    
    r    = np.sqrt( (d * np.cos( b ) )**2 + constants.Rsun * (constants.Rsun - 2 * d * np.cos( b ) * np.cos( l ) ) )
    t    = np.rad2deg( np.arcsin(d * np.sin( l ) * np.cos( b ) / r) )
    z    = constants.Zsun + d * np.sin( b - np.arctan( constants.Zsun / constants.Rsun) )
    
    return r, t, z


########################################################################


def calc_rho(R, Z, rho0 = None):

    '''
    Function to convert ra, dec, and distances into 
    Galactocentric coordinates R, theta, Z.
    '''

    R = np.array(R).flatten()
    Z = np.array(Z).flatten()

    if rho0 is None: 
        rho0 = constants.rho0 # Take the value from the constants file if not provided

    rho_thin  = rho0 * np.exp( -1 * abs( Z - constants.Zsun ) / constants.H_thin )  * \
                np.exp(-1. * ( R - constants.Rsun) / constants.L_thin)
    rho_thick = rho0 * np.exp( -1 * abs( Z - constants.Zsun ) / constants.H_thick ) * \
                np.exp(-1. * ( R - constants.Rsun ) / constants.L_thick )
    rho_halo  = rho0 * ( constants.Rsun / np.sqrt( R ** 2 + ( Z / constants.q ) ** 2) ) ** constants.r_halo

    rho       = constants.f_thin * rho_thin + constants.f_thick * rho_thick + constants.f_halo * rho_halo
    frac      = np.array( [ constants.f_thin * rho_thin, constants.f_thick * rho_thick, constants.f_halo * rho_halo ] ) / rho

    return rho, frac


########################################################################


def calc_uvw(R, theta, Z):

    '''
    Use positions in space to detemine expected <average>
    velocity components in cartesian UVW velocity components
    '''

    # Convert everything to numpy arrays
    R     = np.array(R).flatten()
    theta = np.array(theta).flatten()
    Z     = np.array(Z).flatten()

    # Make arrays with the velocity components
    Rdot = np.zeros(len(Z))
    Tdot = (constants.Rvel - 0.013*abs(Z) - 1.56e-5*Z**2) # will later be converted to Tdot(Z) 
    Zdot = np.zeros(len(Z))                               # typical values for the MW in km/s

    theta = np.deg2rad(theta)  # convert degrees to radians

    Xdot = Rdot * np.cos(theta) - Tdot * np.sin(theta)
    Ydot = -1 * ( Rdot*np.sin(theta) + Tdot * np.cos(theta) )

    return Xdot, Ydot, Zdot


########################################################################


def calc_rtz(R, theta, Z):

    '''
    Use positions in space to detemine expected <average>
    velocity components in Galactocylindrical velocity components
    '''

    # Convert everything to numpy arrays
    R     = np.array( R ).flatten()
    theta = np.array( theta ).flatten()
    Z     = np.array( Z ).flatten()

    # Make arrays with the velocity components
    Rdot = np.zeros( len( Z ) )
    Tdot = ( constants.Rvel - 0.013 * abs( Z ) - 1.56e-5 * Z ** 2) 
    Zdot = np.zeros( len( Z ) )                                    

    return Rdot, Tdot, Zdot


########################################################################


def calc_sigmavel(Z):

    '''
    Function to obtain velocity dispersions (UVW) at a given
    Galactic height

    Values obtained by fitting, either
       sigma = coeff * Z^power or
       sigma = coeff1 * Z * coeff2
    data from Bochanski et al. (2006)
    see ~/sdss/uw/velocity_ellipsoid.pro[.ps] FOR fitting algorithm[fit]
    '''

    #coeff  = np.array([7.085, 3.199, 3.702, 10.383, 1.105, 5.403]) # This was for the power-law fit
    #power  = np.array([0.276, 0.354, 0.307,  0.285, 0.625, 0.309]) # This was for the power-law fit

    """
    # Measured as a linear fuction using Bochanski et al. 2007
    coeff1 = np.array([25.2377322033,  14.7878971614,   13.8771101207,
                       34.4139935894, 15.3784418334, 23.8661980341])
    coeff2 = np.array([0.0262979761654, 0.0284643404931, 0.022023363188,
                       0.0511101810905, 0.079302710092, 0.0242548818891])
    """

    # Measured as a linear fuction using Pineda et al. 2016
    coeff1 = np.array([22.4340996509,  13.9177531905,   10.8484431514,
                       64.0402685036, 39.4069980499, 44.7558405875])

    coeff2 = np.array([0.0372420573173, 0.027242838377, 0.0283471755313,
                       0.0705591437518, 0.0890703940209, 0.0203714200634])

    # convert Z to array for optimization
    Z = np.array(Z).flatten()

    # calculate sigma_vel from the empirical fit
    if len(Z) > 1:
    
        #sigmaa = coeff * np.power.outer(abs(Z), power) # This is the old power law way (See Dhital et al. 2010)
        sigmaa = coeff1 + np.outer(abs(Z), coeff2)     # New linear fit from Pineda et al. 2016
    
    else:
    
        #sigmaa = coeff * abs(Z)**power    # This is the old power law way (See Dhital et al. 2010)
        sigmaa = coeff1 + coeff2 * abs(Z)  # New linear fit from Pineda et al. 2016

    return sigmaa


########################################################################


def gen_gaussian_new(mu, sig1, sig2, sig3, f1, f2, f3):
    
    '''
    Function to make random draws for the UVW velocities based
    on the kinematics of the thin and thick disk, and halo.

    Inputs:
    mu   - average velocity component
    sig1 - standard deviation of the thin disk velocity component
    sig2 - standard deviation of the thick disk velocity component
    sig3 - standard deviation of the halo  velocity component
    f1   - fraction of thin disk stars
    f2   - fraction of thick disk stars
    f3   - fraction of halo stars

    Outputs:
    velocities - drawn velocities
    '''

    mu   = np.array(mu).flatten()
    sig1 = np.array(sig1).flatten()
    sig2 = np.array(sig2).flatten()
    sig3 = np.array(sig3).flatten()
    f1   = np.array(f1).flatten()
    f2   = np.array(f2).flatten()
    f3   = np.array(f3).flatten()
    
    # Generate values based off the halo (largest dispersion, 
    # more numbers are better, but computationally expensive)
    # We chose 5-sigma so we wouldn't pull some crazy value
    x    = np.array([np.linspace(i-(5*j), i+(5*j), 30000) for i,j in zip(mu, sig3)])
    
    # Create the combined probability distribution for the thin and thick disk
    PDF  = f1 * ( 1 / ( np.sqrt( 2 * np.pi ) * sig1 ) * np.exp( -1 * ( ( x.T - mu ) / sig1 ) ** 2 / 2.) ) + \
           f2 * ( 1 / ( np.sqrt( 2 * np.pi ) * sig2 ) * np.exp( -1 * ( ( x.T - mu ) / sig2 ) ** 2 / 2.) ) + \
           f3 * ( 1 / ( np.sqrt( 2 * np.pi ) * sig3 ) * np.exp( -1 * ( ( x.T - mu ) / sig3 ) ** 2 / 2.) )

    # Build the cumulative distribution function
    CDF        = np.cumsum(PDF/np.sum(PDF, axis=0), axis=0) 
    
    # Create the inverse cumulative distribution function
    # We interpolate the probability for whatever value is randomly drawn
    inv_cdf    = np.array([ interpolate.interp1d(i, j) for i,j in zip(CDF.T, x)])

    # Need to get the maximim minimum value for the RNG
    # This will make sure we don't get a number outside the interpolation range
    minVal     = np.max(np.min(CDF.T, axis=1))
    r          = np.random.uniform(low = minVal, high = 1.0, size = len(mu))
    #r          = np.random.rand(len(mu)) # Old way

    velocities = np.array([ inv_cdf[i](j) for i,j in zip(range(0, len(r)), r)])
    
    # return the UVW value
    return velocities


########################################################################


def gal_uvw_pm(U=-9999, V=-9999, W=-9999, ra=-9999, dec=-9999, 
               distance=-9999, plx=-9999, lsr=True):

    '''
    Function to convert UVW velocities into proper motions.
    This is the inverse function of Johnson & Soderblom 1987.

    Inputs:
    U        - U velocities (in km/s)
    V        - V velocities (in km/s)
    W        - W velocities (in km/s)
    ra       - R.A. coordinates (in degrees)
    dec      - Dec. coordinates (in degrees)
    distance - distances (in pc)
    plx      - parallaxes (OPTIONAL)

    Outputs:
    radial velocity (km/s), RA proper motion (mas/yr), DEC proper motion (mas/yr)
    '''
    
    # Local Standard of Rest velocity components
    #lsr_vel = np.array( [-8.5, 13.38, 6.49] )  # Coskunoglu et al. 2011
    #lsr_vel = np.array( [-10, 5.25, 7.17] )    # Dehnen & Binney 1998
    lsr_vel = np.array( [-11.1, 12.24, 7.25] ) # Schonrich, Dehnen & Binney 2010

    # Check if (numpy) arrays
    if isinstance(ra, np.ndarray) == False:

        ra       = np.array(ra).flatten()
        dec      = np.array(dec).flatten()
        U        = np.array(U).flatten()
        V        = np.array(V).flatten()
        W        = np.array(W).flatten()
        distance = np.array(distance).flatten()
        plx      = np.array(plx).flatten()

    if ra.size <= 2:

        ra       = np.array([ra]).flatten()
        dec      = np.array([dec]).flatten()
        U        = np.array([U]).flatten()
        V        = np.array([V]).flatten()
        W        = np.array([W]).flatten()
        distance = np.array([distance]).flatten()
        plx      = np.array([plx]).flatten()
 
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

    k = 4.74047                     # Equivalent of 1 AU / yr in km / s   
    t = np.array( [ [ 0.0548755604,  0.8734370902,  0.4838350155], 
                    [ 0.4941094279, -0.4448296300,  0.7469822445], 
                    [-0.8676661490, -0.1980763734,  0.4559837762] ] )
    #t = np.array( [ [ -0.0548755604, -0.8734370902, -0.4838350155],
    #                [  0.4941094279, -0.4448296300,  0.7469822445],
    #                [ -0.8676661490, -0.1980763734,  0.4559837762] ] )

    for i in range(0, len(vrad)):
        a = np.array( [ [cosa[i]*cosd[i], -sina[i], -cosa[i]*sind[i] ], 
                        [sina[i]*cosd[i],  cosa[i], -sina[i]*sind[i] ], 
                        [sind[i]        ,  0      ,  cosd[i]         ] ] )
        b = np.dot(t, a)

        uvw = np.array([ U[i], V[i], W[i] ])

        # Correct for stellar motion
        if lsr == True:
            uvw = uvw - lsr_vel

        #vec = np.dot( np.transpose(uvw), b)
        vec = np.dot( np.transpose(b), uvw)

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


########################################################################


def gen_pm1(R0, T0, Z0, ra0, dec0, dist0, UVW = False):

    '''
    Old function to determine UVW velocities.
    This uses distpersions in UVW to build velocity
    component distribution.
    '''

    sigmaa    = calc_sigmavel(Z0)                                                    # calculate the UVW velocity dispersions                                                                                 # returns [U_thin,V_thin,W_thin,U_thick,V_thick,W_thick]
    rho, frac = calc_rho(R0, Z0)                                                     # calc the frac of thin/thick disk stars                                                                            # returns frac = [f_thin, f_thick, f_halo]
    vel       = np.array(calc_uvw(R0, T0, Z0)) - \
                np.array(calc_uvw(constants.Rsun, constants.Tsun, constants.Zsun))   # convert to cartesian velocities            
                                                                                     # returns [U,V,W]

    # Convert velocities and dispersions to UVW velocities. 
    # Halo values taken from minimum value of Bond et al. 2010.
    if len(R0) == 1:
    
        U = gen_gaussian_new(vel[0], sigmaa[0], sigmaa[3], np.zeros(len(frac[0]))+135., frac[0], frac[1], frac[2])
        V = gen_gaussian_new(vel[1], sigmaa[1], sigmaa[4], np.zeros(len(frac[0]))+85.,  frac[0], frac[1], frac[2])
        W = gen_gaussian_new(vel[2], sigmaa[2], sigmaa[5], np.zeros(len(frac[0]))+85.,  frac[0], frac[1], frac[2])
    
    else:

        U = gen_gaussian_new(vel[0], sigmaa[:,0], sigmaa[:,3], np.zeros(len(frac[0]))+135., frac[0], frac[1], frac[2])
        V = gen_gaussian_new(vel[1], sigmaa[:,1], sigmaa[:,4], np.zeros(len(frac[0]))+85.,  frac[0], frac[1], frac[2])
        W = gen_gaussian_new(vel[2], sigmaa[:,2], sigmaa[:,5], np.zeros(len(frac[0]))+85.,  frac[0], frac[1], frac[2])

    # change UVW to pmra and pmdec
    rv, pmra, pmdec = gal_uvw_pm(U = U, V = V, W = W, ra = ra0,
                                 dec = dec0, distance = dist0 )

    if UVW is True: 
        return pmra, pmdec, rv, U, V, W

    else: 
        return pmra, pmdec, rv


########################################################################


def gen_pm2(R0, T0, Z0, ra0, dec0, dist0, UVW = False):

    '''
    New function to determine UVW velocities.
    This uses distpersions in UVW to build velocity
    component distribution.
    '''

    sigmaa    = calc_sigmavel(Z0)                                                    # calculate the RTZ velocity dispersions                                                                                 # returns [R_thin,T_thin,Z_thin,R_thick,T_thick,Z_thick]
    rho, frac = calc_rho(R0, Z0)                                                     # calc the frac of thin/thick disk stars                                                                            # returns frac = [f_thin, f_thick, f_halo]
    vel       = np.array(calc_rtz(R0, T0, Z0)) - \
                np.array(calc_rtz(constants.Rsun, constants.Tsun, constants.Zsun))   # convert to galactocylindrical velocities            
                                                                                     # returns [R,T,Z]

    # Convert velocities and dispersions to RTZ velocities. 
    # Halo values taken from minimum value of Bond et al. 2010.
    if len(R0) == 1:

        Rdot = gen_gaussian_new(vel[0], sigmaa[0], sigmaa[3], np.zeros(len(frac[0]))+135., frac[0], frac[1], frac[2])
        Tdot = gen_gaussian_new(vel[1], sigmaa[1], sigmaa[4], np.zeros(len(frac[0]))+85.,  frac[0], frac[1], frac[2])
        Zdot = gen_gaussian_new(vel[2], sigmaa[2], sigmaa[5], np.zeros(len(frac[0]))+85.,  frac[0], frac[1], frac[2])
    
    else:

        Rdot = gen_gaussian_new(vel[0], sigmaa[:,0], sigmaa[:,3], np.zeros(len(frac[0]))+135., frac[0], frac[1], frac[2])
        Tdot = gen_gaussian_new(vel[1], sigmaa[:,1], sigmaa[:,4], np.zeros(len(frac[0]))+85.,  frac[0], frac[1], frac[2])
        Zdot = gen_gaussian_new(vel[2], sigmaa[:,2], sigmaa[:,5], np.zeros(len(frac[0]))+85.,  frac[0], frac[1], frac[2])

    # change to UVW
    theta = np.deg2rad(T0)  # convert degrees to radians
    U = Tdot * np.sin(theta) - Rdot * np.cos(theta)  # Source of contention (positive or negative)
    #U = Tdot * np.sin(theta) + Rdot * np.cos(theta) 
    V = Tdot * np.cos(theta) - Rdot * np.sin(theta) 
    W = Zdot

    # change UVW to pmra and pmdec
    rv, pmra, pmdec = gal_uvw_pm(U = U, V = V, W = W, ra = ra0,
                                 dec = dec0, distance = dist0 )

    if UVW == True: 
        return pmra, pmdec, rv, U, V, W
    else: 
        return pmra, pmdec, rv


########################################################################


def inverse_transform_sampling(nstars, dists, n_samples):

    '''
    Function to do sampling from the inverse
    cumulative distribution function.
    '''

    #data = np.load('Distance_Dist.npy')
    #n_bins = int(np.sqrt(len(data)))
    #hist, bin_edges = np.histogram(data, bins=n_bins, density=True) 
    hist, bin_edges = nstars / np.trapz( y=nstars, x=dists ), np.append( dists, dists[-1] + np.diff( dists )[0] )   
    cum_values      = np.zeros(bin_edges.shape)
    cum_values[1:]  = np.cumsum( hist*np.diff( bin_edges ) )
    inv_cdf         = interpolate.interp1d( cum_values, bin_edges )
    r               = np.random.rand( int( n_samples ) )

    return inv_cdf(r)


########################################################################


def gen_nstars(ra0, dec0, num, nstars, dists, cellsize = None, range1 = False, size = 10000):

    '''
    Function to generate random positions and distances 
    for stars based on a given line of sight.

    Inputs:
    ra0     - Input central R.A. (in degrees)
    dec0    - Input central Dec. (in degrees)
    num     - Number of stars to simluate. This can be a float
              and an integer will be assigned based on probabilty.
    nstars  - number of stars per distance bin along the LOS
    dists   - array of distances pertaining to nstars

    Outputs (one for each simulated star, returned as arrays): 
    R.A. (in degrees), Dec (in degrees, Distance (in pc) 

    Optional Outputs (used for inverse sampling):
    nstars  - number of stars per distance bin along the LOS
    dists   - array of distances pertaining to nstars 
    '''
    
    # Get the cell size, or use the default of 30' x 30' 
    if cellsize is None: 
        cellsize = constants.cell_size

    # Check if (numpy) arrays
    if isinstance( ra0, np.ndarray )  == False:
        ra0  = np.array( [ra0] ).flatten()
    if isinstance( dec0, np.ndarray) == False:
        dec0 = np.array( [dec0] ).flatten()
    if isinstance( num, np.ndarray )  == False:
        num  = np.array( [num] ).flatten()

    if len(num) != 1:
        raise ValueError('Number of stars (%s) must be a single value'%num)

    # Check if ranges or not
    if len(ra0) == 2 and len(dec0) == 2:
        range1 = True

    # Check if number a float instead of an array
    if isinstance(num[0], float): # Need to generate an array of random numbers based off number probability
        randstar    = np.random.random_sample( size )
        nstars_tot2 = np.zeros( size, dtype=np.int64 ) + np.floor( num[0] )
        nstars_tot2[ np.where( randstar < num[0] - np.floor( num[0] ) ) ] += 1
        nstars_tot  = np.random.choice(nstars_tot2, size = 1) # Select a random choice of stars (need an integer)
    elif isinstance(num[0], int):
        nstars_tot  = num[0]
    else:
        raise ValueError('Number of stars (%s) must be an integer or float'%num)
    
    # Randomly assign RADEC values.
    # This may not be 100% correct since absolute placement is probably 
    # favored at lower Galactic latitudes
    if range1 is True: # Case for ranges
        ra1   = np.random.uniform( min(ra0),  max(ra0),  size = nstars_tot )
        dec1  = np.random.uniform( min(dec0), max(dec0), size = nstars_tot )
    else: # Case for cell size
        ra1   = ra0  + ( ( np.random.rand( int( nstars_tot ), 1 ).flatten() - 0.5 ) * cellsize )
        dec1  = dec0 + ( ( np.random.rand( int( nstars_tot ), 1 ).flatten() - 0.5 ) * cellsize )

    # Pull distances from a defined distribution
    dist1     = inverse_transform_sampling( nstars, dists, nstars_tot )  

    return ra1, dec1, dist1


########################################################################


def radec2lb(ra, dec):

    '''
    Convert ra,dec values into Galactic l,b coordinates
    '''

    # Make numpy arrays
    if isinstance(ra, np.ndarray)  == False:
        ra = np.array(ra).flatten()
    if isinstance(dec, np.ndarray) == False:
        dec = np.array(dec).flatten()
        
    # Fix the dec values if needed, should be between (-90,90)
    dec[ np.where( dec > 90 )]  = dec[ np.where( dec > 90 )]  - 180
    dec[ np.where( dec < -90 )] = dec[ np.where( dec < -90 )] + 180
    
    psi    = 0.57477043300
    stheta = 0.88998808748
    ctheta = 0.45598377618
    phi    = 4.9368292465
    
    a    = np.deg2rad( ra ) - phi
    b    = np.deg2rad( dec )
    sb   = np.sin( b )
    cb   = np.cos( b )
    cbsa = cb * np.sin( a )
    b    = -1 * stheta * cbsa + ctheta * sb
    bo   = np.rad2deg( np.arcsin( np.minimum(b, 1.0) ) )
    del b
    a    = np.arctan2( ctheta * cbsa + stheta * sb, cb * np.cos( a ) )
    del cb, cbsa, sb
    ao   = np.rad2deg( ( (a + psi + 4. * np.pi ) % ( 2. * np.pi ) ) )

    return ao, bo


