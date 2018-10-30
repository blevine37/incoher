import numpy as np
import math
import incoher.dm as dm

nk = 10

omega = .01

mass = 2000

k = mass * omega * omega

deltae = 0.20

c = -0.2

dt = 1.0

intensity = 1.0

rho = dm(nk,k,deltae,c,mass,dt,intensity)

rho.init_0k()

print rho.rho00r
print rho.rho00i
print rho.omega

rho.build_H()
rho.build_mu()

print rho.H00
print rho.H11
print rho.mu01
print rho.mu10

rho.init_0k_excited()

rho.compute_rhodot_r_H()
rho.compute_rhodot_i_H()

print rho.rhodot00r_H
print rho.rhodot00i_H
print rho.rhodot11r_H
print rho.rhodot11i_H
