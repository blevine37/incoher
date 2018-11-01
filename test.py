import numpy as np
import math
import incoher.dm as dm

np.random.seed(25892)

nruns = 1000

nk = 2

omega = .01

mass = 2000

k = mass * omega * omega

deltae = 0.20

b = -0.2
#b = 0.0

dt = 1.0

intensity = 1.0e-3

rho = dm(nk,k,deltae,b,mass,dt,intensity)

rhobar = dm(nk,k,deltae,b,mass,dt,intensity)

rhobar.open_output_file("rhobar.hdf5")

for irun in range(nruns):

    rho.open_output_file("rho.hdf5")

    rho.init_0k()

    rho.prop_whitenoise(10000,100)

    rhobar.add_dm_to_average(rho)

    rho.close_output_file()

rhobar.close_output_file()

