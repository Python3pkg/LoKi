class constants(object):

	def __init__(self):

		# Stellar Position Parameters
		self.Rsun      = 8500.                  # R coord for the Sun (in pc)
		self.Tsun      = 0.                     # theta coord for the Sun 
		self.Zsun      = 15.                    # Z coord for the Sun (in pc)

		# Rotation Curve Velocity
		self.Rvel      = 240.                   # Galactic rotation curve velocity (km/s)  

		# Defaults
		self.rho0      = 0.03306                # rho0 from integrating the Bochanski et al. (2010) corrected LF   
		self.cell_size = 0.5                    # size of one cell (in degrees)
		self.max_dist  = 1000.                  # maximum allowed distance for simulated star (in pc)
		self.min_dist  = 0.                     # minimum allowed distance for simulated star (in pc)
		self.ddist     = 1.                     # steps in distance (in pc)

		# Density Profile Parameters
		self.H_thin    = 300.                   # scale height for the thin disk (in pc)
		self.H_thick   = 2100.                  # scale height for the thick disk (in pc)
	    
		self.L_thin    = 3100.                  # scale length for the thin disk (in pc)
		self.L_thick   = 3700.                  # scale length for the thick disk (in pc)
	    
		self.f_thick   = 0.04                   # fraction of thick disk stars in the solar neighborhood
		self.f_halo    = 0.0025                 # fraction of halo stars in the solar neighborhood
	    
		self.f_thin    = 1. - self.f_thick - self.f_halo  # fraction of thin disk stars in the solar neighborhood
		self.r_halo    = 2.77                   # halo density gradient
		self.q         = 0.64                   # halo flattening parameter
