import numpy as np
import math
import incoher.dm as dm

nk = 10

rho = dm(nk)

print rho.rho00
print rho.rho01
print rho.rho10
print rho.rho11

rho.init_0k()

print rho.rho00
