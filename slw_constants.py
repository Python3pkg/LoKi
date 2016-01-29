Rsun      = 8500.            # R coord for the Sun in parsecs
Tsun      = 0.               # theta coord for the Sun 
Zsun      = 15.              # Z coord for the Sun in parsecs
rho0      = 0.03306          # rho0 from integrating the Bochanski et al. (2010) corrected LF     
au2pc     = 1 / 206264.806   # conversion from AU to parsecs
cell_size = 0.5              # size of one cell in degrees
max_dist  = 1000.            # maximum allowed distance for simulated star
min_dist  = 0.               # minimum allowed distance for simulated star

# Density Profile Parameters
H_thin    = 300.             # scale heights in pc
H_thick   = 2100.            # scale heights in pc
    
L_thin    = 3100.            # scale lengths in pc
L_thick   = 3700.            # scale lengths in pc
    
f_thick   = 0.04             # fraction of thick disk stars in the solar neighborhood
f_halo    = 0.0025           # fraction of halo stars in the solar neighborhood
    
f_thin    = 1. - f_thick - f_halo  # fraction of thin disk stars in the solar neighborhood
r_halo    = 2.77                   # halo density gradient
q         = 0.64                   # halo flattening parameter