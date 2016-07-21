# Stellar Position Parameters
Rsun      = 8500.                  # R coord for the Sun (in pc)
Tsun      = 0.                     # theta coord for the Sun 
Zsun      = 15.                    # Z coord for the Sun (in pc)

# Rotation Curve Velocity
Rvel      = 240.                   # Galactic rotation curve velocity (km/s)  

# Defaults
rho0      = 0.03306                # rho0 from integrating the Bochanski et al. (2010) corrected LF   
cell_size = 0.5                    # size of one cell (in degrees)
max_dist  = 1000.                  # maximum allowed distance for simulated star (in pc)
min_dist  = 0.                     # minimum allowed distance for simulated star (in pc)
ddist     = 1.                     # steps in distance (in pc)

# Density Profile Parameters
H_thin    = 300.                   # scale height for the thin disk (in pc)
H_thick   = 2100.                  # scale height for the thick disk (in pc)
    
L_thin    = 3100.                  # scale length for the thin disk (in pc)
L_thick   = 3700.                  # scale length for the thick disk (in pc)
    
f_thick   = 0.04                   # fraction of thick disk stars in the solar neighborhood
f_halo    = 0.0025                 # fraction of halo stars in the solar neighborhood
    
f_thin    = 1. - f_thick - f_halo  # fraction of thin disk stars in the solar neighborhood
r_halo    = 2.77                   # halo density gradient
q         = 0.64                   # halo flattening parameter
