import numpy as np
import time, os
import random
import scipy.interpolate as interpolate
import slw_constants as slw_c
import matplotlib.pyplot as plt

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

    R     = np.array(R).flatten()
    theta = np.array(theta).flatten()
    Z     = np.array(Z).flatten()

    Rdot = np.zeros(len(Z))
    Tdot = (240. - 0.013*abs(Z) - 1.56e-5*Z**2) #/ R    # will later be converted to Tdot(Z) 
    Zdot = np.zeros(len(Z))                             # typical values for the MW in km/s

    theta = np.deg2rad(theta)  # convert degrees to radians

    Xdot = Rdot * np.cos(theta) - Tdot * np.sin(theta)
    Ydot = -1 * ( Rdot*np.sin(theta) + Tdot * np.cos(theta) )

    return Xdot, Ydot, Zdot

########################

def calc_rtz(R, theta, Z):

    R     = np.array(R).flatten()
    theta = np.array(theta).flatten()
    Z     = np.array(Z).flatten()

    Rdot = np.zeros(len(Z))
    Tdot = (240. - 0.013*abs(Z) - 1.56e-5*Z**2) #/ R    # will later be converted to Tdot(Z) 
    Zdot = np.zeros(len(Z))                             # typical values for the MW in km/s

    return Rdot, Tdot, Zdot

########################

def calc_sigmavel(Z):

    # U_thin,V_thin,W_thin,U_thick,V_thick,W_thick
    # Values obtained by fitting sigma = coeff * Z^power 
    # data from Bochanski et al. (2006)
    # see ~/sdss/uw/velocity_ellipsoid.pro[.ps] FOR fitting algorithm[fit]
    coeff  = np.array([7.085, 3.199, 3.702, 10.383, 1.105, 5.403]) # This was for the power-law fit
    power  = np.array([0.276, 0.354, 0.307,  0.285, 0.625, 0.309]) # This was for the power-law fit
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

    # calculate sigma_vel from the empirical power-law fit
    if len(Z) > 1:
        #sigmaa = coeff * np.power.outer(abs(Z), power)
        #print np.power.outer(abs(Z), power).shape
        #sys.exit()
        sigmaa = coeff1 + np.outer(abs(Z), coeff2)
    else:
        #sigmaa = coeff * abs(Z)**power
        sigmaa = coeff1 + coeff2 * abs(Z)

    return sigmaa

########################

def calc_rho(R, Z, rho0 = None):

    R = np.array(R).flatten()
    Z = np.array(Z).flatten()

    if rho0 == None: rho0 = slw_c.rho0 # Take the value from the constants file if not provided

    rho_thin  = rho0 * np.exp(-1. * abs(Z-slw_c.Zsun) / slw_c.H_thin)  * np.exp(-1. * (R-slw_c.Rsun) / slw_c.L_thin)
    rho_thick = rho0 * np.exp(-1. * abs(Z-slw_c.Zsun) / slw_c.H_thick) * np.exp(-1. * (R-slw_c.Rsun) / slw_c.L_thick)
    rho_halo  = rho0 * ( slw_c.Rsun / np.sqrt( R**2 + ( Z / slw_c.q )**2) )**slw_c.r_halo

    rho       = slw_c.f_thin * rho_thin + slw_c.f_thick * rho_thick + slw_c.f_halo * rho_halo
    frac      = np.array([slw_c.f_thin * rho_thin, slw_c.f_thick * rho_thick, slw_c.f_halo * rho_halo]) / rho

    return rho, frac

########################

def gen_gaussian_new(mu, sig1, sig2, sig3, f1, f2, f3):
    
    # We use this to make random draws for the UVW velocities based on the kinematics
    # of the thin and thick disk

    mu   = np.array(mu).flatten()
    sig1 = np.array(sig1).flatten()
    sig2 = np.array(sig2).flatten()
    sig3 = np.array(sig3).flatten()
    f1   = np.array(f1).flatten()
    f2   = np.array(f2).flatten()
    f3   = np.array(f3).flatten()
    
    # Generate values based off the halo (largst dispersion, more numbers the better, but computationally expensive)
    # We chose 5-sigma so we wouldn't pull some crazy value
    x = np.array([np.linspace(i-(5*j), i+(5*j), 30000) for i,j in zip(mu, sig3)])
    
    # Create the combined probability distribution for the thin and thick disk
    PDF = f1*(1 / (np.sqrt(2*np.pi) * sig1)*np.exp(-((x.T-mu)/sig1)**2 / 2)) + \
          f2*(1 / (np.sqrt(2*np.pi) * sig2)*np.exp(-((x.T-mu)/sig2)**2 / 2)) + \
          f3*(1 / (np.sqrt(2*np.pi) * sig3)*np.exp(-((x.T-mu)/sig3)**2 / 2))

    # Build the cumulative distribution function
    CDF = np.cumsum(PDF/np.sum(PDF, axis=0), axis=0) 
    
    # Create the inverse cumulative distribution function
    # We interpolate the probability for whatever value is randomly drawn
    inv_cdf = np.array([ interpolate.interp1d(i, j) for i,j in zip(CDF.T, x)])

    # Need to get the maximim minimum value for the RNG
    # This will make sure we don't get a number outside the interpolation range
    minVal = np.max(np.min(CDF.T, axis=1))
    r = np.random.uniform(low = minVal, high = 1.0, size = len(mu))
    #r = np.random.rand(len(mu)) # Old way

    results = np.array([ inv_cdf[i](j) for i,j in zip(range(0, len(r)), r)])
    
    # return the UVW value
    return results

########################

def gal_uvw_pm(U=-9999, V=-9999, W=-9999, ra=-9999, dec=-9999, distance=-9999, plx=-9999, lsr=True):

    
    #lsr_vel = np.array([-8.5, 13.38, 6.49])  # Coskunoglu et al. 2011
    #lsr_vel = np.array([-10, 5.25, 7.17])    # Dehnen & Binney 1998
    lsr_vel = np.array([-11.1, 12.24, 7.25]) # Schonrich, Dehnen & Binney 2010
    #lsr_vel = np.array([-10.3, 6.3, 5.9]) # Test

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

########################

def gen_pm_new(R0, T0, Z0, ra0, dec0, dist0, test=False):

    sigmaa    = calc_sigmavel(Z0)                                                       # calculate the UVW velocity dispersions                                                                                 # returns [U_thin,V_thin,W_thin,U_thick,V_thick,W_thick]
    rho, frac = calc_rho(R0, Z0)                                                        # calc the frac of thin/thick disk stars                                                                            # returns frac = [f_thin, f_thick, f_halo]
    vel       = np.array(calc_uvw(R0, T0, Z0)) - np.array(calc_uvw(slw_c.Rsun, slw_c.Tsun, slw_c.Zsun))   # convert to cartesian velocities            
                                                                                        # returns [U,V,W]

    # Convert velocities and dispersions to UVW velocities. Halo values taken from minimum value of Bond et al. 2010.
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

    if test == True: return U, V, W
    else: return pmra, pmdec, rv

########################

def gen_pm_new2(R0, T0, Z0, ra0, dec0, dist0, test=False):

    sigmaa    = calc_sigmavel(Z0)                                                       # calculate the RTZ velocity dispersions                                                                                 # returns [R_thin,T_thin,Z_thin,R_thick,T_thick,Z_thick]
    rho, frac = calc_rho(R0, Z0)                                                        # calc the frac of thin/thick disk stars                                                                            # returns frac = [f_thin, f_thick, f_halo]
    vel       = np.array(calc_rtz(R0, T0, Z0)) - np.array(calc_rtz(slw_c.Rsun, slw_c.Tsun, slw_c.Zsun))   # convert to galactocylindrical velocities            
                                                                                        # returns [R,T,Z]

    # Convert velocities and dispersions to RTZ velocities. Halo values taken from minimum value of Bond et al. 2010.
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
    U = -1 * ( Rdot * np.cos(theta) + Tdot * np.sin(theta) ) # Source of contention
    #U = Rdot * np.cos(theta) + Tdot * np.sin(theta) 
    V = Tdot * np.cos(theta) - Rdot * np.sin(theta) 
    W = Zdot

    # change UVW to pmra and pmdec
    rv, pmra, pmdec = gal_uvw_pm(U = U, V = V, W = W, ra = ra0,
                                 dec = dec0, distance = dist0 )

    if test == True: return U, V, W
    else: return pmra, pmdec, rv

########################

def inverse_transform_sampling(nstars, dists, n_samples):
    #data = np.load('Distance_Dist.npy')
    #n_bins = int(np.sqrt(len(data)))
    #hist, bin_edges = np.histogram(data, bins=n_bins, density=True) 
    hist, bin_edges = nstars/np.trapz(y=nstars, x=dists), np.append(dists, dists[-1]+np.diff(dists)[0])   
    cum_values = np.zeros(bin_edges.shape)
    cum_values[1:] = np.cumsum(hist*np.diff(bin_edges))
    inv_cdf = interpolate.interp1d(cum_values, bin_edges)
    r = np.random.rand(int(n_samples))
    return inv_cdf(r)
    
########################

def conv_to_galactic(ra, dec, d):

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
    c_icrs = SkyCoord(ra = ra*u.degree, dec = dec*u.degree, frame = 'icrs')  # The SLOW Astropy way
    l, b = c_icrs.galactic.l.radian, c_icrs.galactic.b.radian
    """
    l, b = radec2lb(ra, dec)
    l, b = np.deg2rad(l), np.deg2rad(b)
    
    r   = np.sqrt( (d * np.cos( b ) )**2 + slw_c.Rsun * (slw_c.Rsun - 2 * d * np.cos( b ) * np.cos( l ) ) )
    t   = np.rad2deg( np.arcsin(d * np.sin( l ) * np.cos( b ) / r) )
    z   = slw_c.Zsun + d * np.sin( b - np.arctan( slw_c.Zsun / slw_c.Rsun) )
    
    return r, t, z

########################

def gen_nstars(ra0, dec0, num, nstars, dists, cellsize = None):

    # ra0, dec0, num - input parameters
    # ra, dec, dist  - output arrays for the generated stars
    # nstars         - number of stars per distance bin along the LOS
    # dists          - array of distances pertaining to nstars
    
    # Get the cell size, or use the default of 30' x 30' 
    if cellsize == None:
        cellsize = slw_c.cell_size

    # Check if (numpy) arrays
    if isinstance(ra0, np.ndarray) == False:
        ra0  = np.array(ra0)
        dec0 = np.array(dec0)

    n_acc = 0                # number of stars accepted

    while n_acc < num:
    
        # Seed the random generator each time
        rand = int(os.urandom(4).encode('hex'), 16)
        np.random.seed(rand)
    
        n_lft     = num - n_acc # no. of needed stars

        ra1       = ra0  + ((np.random.rand(n_lft, 1).flatten() - 0.5) * cellsize)
        dec1      = dec0 + ((np.random.rand(n_lft, 1).flatten() - 0.5) * cellsize)
        #dist1     = np.random.rand(n_lft,1).flatten() * slw_c.max_dist
        #dist1     = np.random.uniform(low=slw_c.min_dist, high=slw_c.max_dist, size=(n_lft,1)).flatten() # This is correct
        #dist1     = inverse_transform_sampling(n_lft)  # This pulls from a defined distribution
        #dist1     = np.random.power(4, n_lft) * slw_c.max_dist  # This pulls from a power-law distribution
        dist1     = inverse_transform_sampling(nstars, dists, n_lft)  # This pulls from a defined distribution

        R, T, Z   = conv_to_galactic(ra1, dec1, dist1)

        rho, frac = calc_rho(R, Z)

        rand1     = np.random.rand(n_lft, 1).flatten()

        # accept if random number is less than rho(R,Z)/rho0
        ind       = np.where(rand1 < rho / slw_c.Zsun)
        count     = len(ind[0])

        if count != 0:
            if n_acc == 0:
                ra   = ra1[ind]
                dec  = dec1[ind]
                dist = dist1[ind]
            else:
                ra   = np.append(ra, ra1[ind].flatten())
                dec  = np.append(dec, dec1[ind].flatten())
                dist = np.append(dist, dist1[ind].flatten())
            n_acc += count

    return ra, dec, dist

########################

def gen_nstars_new(ra0, dec0, num, nstars, dists, cellsize = None, range1=False):

    # ra0, dec0, num - input parameters
    # ra, dec, dist  - output arrays for the generated stars
    # nstars         - number of stars per distance bin along the LOS
    # dists          - array of distances pertaining to nstars
    
    # Get the cell size, or use the default of 30' x 30' 
    if cellsize == None: cellsize = slw_c.cell_size

    # Check if (numpy) arrays
    if isinstance(ra0, np.ndarray) == False:
        ra0  = np.array([ra0]).flatten()

    if isinstance(dec0, np.ndarray) == False:
        dec0 = np.array([dec0]).flatten()

    if len(ra0) == 2:
        range1 = True
    
    if range1 == True:
        ra1  = np.random.uniform( min(ra0), max(ra0), size=num)
        dec1 = np.random.uniform( min(dec0), max(dec0), size=num)
    else:
        ra1       = ra0  + ((np.random.rand(int(num), 1).flatten() - 0.5) * cellsize)
        dec1      = dec0 + ((np.random.rand(int(num), 1).flatten() - 0.5) * cellsize)
    dist1     = inverse_transform_sampling(nstars, dists, num)  # This pulls from a defined distribution

    return ra1, dec1, dist1

########################

def count_nstars(ra, dec, rho0 = None, cellsize = None, number = False, maxdist = None, mindist = None, range1 = False):

    # Check if RADEC are arrays, and if so sort them
    if isinstance(ra, np.ndarray) == False:
        ra = np.array([ra]).flatten()
        ra = np.sort(ra)
    if isinstance(dec, np.ndarray) == False:
        dec = np.array([dec]).flatten()
        dec = np.sort(dec)
        
    ddist = 1.                                      # steps in distance in pc

    # Grab the distance limits from the file if non given
    if mindist == None:
        mindist = slw_c.min_dist
    if maxdist == None:
        maxdist = slw_c.max_dist

    dist    = np.arange(mindist, maxdist+ddist, ddist, dtype=np.float)   # min_dist < d < max_dist in X pc steps
    n       = len(dist)                                                  # number of steps to take
    deg2rad = np.pi / 180.                                               # Convert degrees to radians

    # If the density isn't set, get it from the constants file
    if rho0 == None: 
        rho0 = slw_c.rho0
    
    # Get the cell size, or use the default of 30' x 30' 
    if cellsize == None: 
        cellsize = slw_c.cell_size
    
    # create an array to store rho for each d
    rho    = np.empty(n, dtype=np.float) 
    nstars = np.empty(n, dtype=np.float) 
 
    # define fractional positions so that rho can be averaged
    x = np.array([-0.5,  0.0,  0.5,  -0.25,  0.25, -0.5,
                   0.0,  0.5, -0.25,  0.25, -0.5,   0.0,  0.5]) * cellsize
    y = np.array([-0.5, -0.5, -0.5,  -0.25, -0.25,  0.0,
                   0.0,  0.0,  0.25,  0.25,  0.5,   0.5,  0.5]) * cellsize

    if len(ra) == 2:
        range1 = True
        nx, ny = (100, 100)
        xv     = np.linspace(ra[0], ra[1], nx)
        yv     = np.linspace(dec[0], dec[1], ny)
        x, y   = np.meshgrid(xv, yv)

    for k in range(0, len(dist)): # step through each distance to integrate out

        if range1 == True:
            R, T, Z       = conv_to_galactic( x, y, dist[k] )                 # Convert coordinates to galactic cylindrical
        else:
            R, T, Z       = conv_to_galactic( ra+x, dec+y, dist[k] )          # Convert coordinates to galactic cylindrical
        
        rhoTemp, frac = calc_rho( R, Z, rho0=rho0 )                           # Calculate the stellar density
        rho[k]        = np.mean( rhoTemp )                                    # Take the mean density across the volume
        
        if range1 == True:
            #vol = 1./3. * ((dist[k])**3 - (dist[k]+ddist)**3) * (np.cos(max(dec)*d2r) - np.cos(min(dec)*d2r)) * (max(ra) - min(ra))*d2r
            #vol = 2.*np.tan( .5*(max(ra)-min(ra))*deg2rad ) * 2.*np.tan( .5*(max(dec)-min(dec))*deg2rad ) * np.cos( (dec[0]+dec[1]) / 2. * deg2rad) * (dist[k] + ddist/2.)**2 * (ddist)
            vol  = 1./3. * ((dist[k] + ddist)**3 - (dist[k])**3) * (np.cos( (90-dec[0])*deg2rad ) - np.cos( (90-dec[1])*deg2rad) ) * (ra[0] - ra[1])*deg2rad
            if vol < 0: vol = abs(vol)
        else: # The case for a gridsize
            #sideRA        = cellsize * deg2rad * np.cos(dec*deg2rad)                 # cellsize in radians (small angle approximation)
            #sideDEC       = cellsize * deg2rad                                       # cellsize in radians (small angle approximation)
            #sideRA        = 2.*np.tan( .5*cellsize*deg2rad ) * np.cos(dec*deg2rad)   # cellsize in radians (full solution)
            #sideDEC       = 2.*np.tan( .5*cellsize*deg2rad )                         # cellsize in radians (full solution)
            #vol           = (sideRA * sideDEC) * (dist[k]+ddist/2.)**2 * ddist   # volume at that d = d * d * h
            vol  = 1./3. * ((dist[k] + ddist)**3 - (dist[k])**3) * (np.cos( (90-dec-cellsize/2.)*deg2rad ) - np.cos( (90-dec+cellsize/2.)*deg2rad) ) * ((ra+cellsize/2.) - (ra-cellsize/2.))*deg2rad
            if vol < 0: vol = abs(vol)
        nstars[k]     = rho[k] * vol                                             # Add the number of stars (density * volume)

    nstars_tot = np.sum(nstars)                                                  # Compute the total number of stars within the volume
    
    if number == True: # This says you want the distributions of stars and distances
        return nstars_tot, nstars, dist+ddist/2.
    else: 
        return nstars_tot

########################

def radec2lb(ra, dec):

    from numpy import array, sin, cos, pi, deg2rad, rad2deg, arctan2, arcsin, minimum

    # Make numpy arrays
    if isinstance(ra, np.ndarray) == False:
        ra = np.array(ra).flatten()
    if isinstance(dec, np.ndarray) == False:
        dec = np.array(dec).flatten()
        
    # Fix the dec values if needed, should be between (-90,90)
    dec[np.where(dec > 90)]  = dec[np.where(dec > 90)] - 180
    dec[np.where(dec < -90)] = dec[np.where(dec < -90)] + 180
    
    psi    = 0.57477043300
    stheta = 0.88998808748
    ctheta = 0.45598377618
    phi    = 4.9368292465
    
    a = deg2rad(ra) - phi
    b = deg2rad(dec)
    sb = sin(b)
    cb = cos(b)
    cbsa = cb * sin(a)
    b = -stheta * cbsa + ctheta * sb
    bo = rad2deg(arcsin(minimum(b, 1.0)))
    del b
    a = arctan2(ctheta * cbsa + stheta * sb, cb * cos(a))
    del cb, cbsa, sb
    ao = rad2deg(((a + psi + 4 * pi ) % (2 * pi) ) )

    return ao, bo

########################
