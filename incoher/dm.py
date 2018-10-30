import numpy as np
import math

# PESs:
# V0 = 1/2 k x^2
# V1 = 1/2 k x^2 + b x + deltae

class dm(object):
    def __init__(self,nk,k,deltae,b,mass,dt,intensity):
        self.nk = nk
        self.k = k
        self.deltae = deltae
        self.b = b
        self.mass = mass
        self.omega = math.sqrt(k/mass)
        self.intensity = instensity
        self.dt = dt

    def init_0k(self):
        self.rho00r = np.zeros((self.nk,self.nk))
        self.rho01r = np.zeros((self.nk,self.nk))
        self.rho10r = np.zeros((self.nk,self.nk))
        self.rho11r = np.zeros((self.nk,self.nk))
        self.rho00i = np.zeros((self.nk,self.nk))
        self.rho01i = np.zeros((self.nk,self.nk))
        self.rho10i = np.zeros((self.nk,self.nk))
        self.rho11i = np.zeros((self.nk,self.nk))
        self.rho00r[0,0] = 1.0

    def init_0k_excited(self):
        self.rho00r = np.zeros((self.nk,self.nk))
        self.rho01r = np.zeros((self.nk,self.nk))
        self.rho10r = np.zeros((self.nk,self.nk))
        self.rho11r = np.zeros((self.nk,self.nk))
        self.rho00i = np.zeros((self.nk,self.nk))
        self.rho01i = np.zeros((self.nk,self.nk))
        self.rho10i = np.zeros((self.nk,self.nk))
        self.rho11i = np.zeros((self.nk,self.nk))
        self.rho11r[0,0] = 1.0

    def build_H(self):
        self.H00 = np.zeros((self.nk,self.nk))
        self.H11 = np.zeros((self.nk,self.nk))
        #diagonals
        for i in range(self.nk):
            self.H00[i,i] = self.omega * (i + 0.5)
            self.H11[i,i] = self.omega * (i + 0.5) + self.deltae
        #off diagonals
        for i in range(self.nk-1):
            # transition dipoles from http://farside.ph.utexas.edu/teaching/qmech/Quantum/node120.html
            tmp = math.sqrt((i+1)/(2.0*self.mass*self.omega))
            self.H11[i,i+1] = self.b * tmp
            self.H11[i+1,i] = self.H11[i,i+1]

    def build_mu(self):
        self.mu01 = np.identity(self.nk)
        self.mu10 = np.identity(self.nk)

    def compute_rhodot_r_H(self):
        self.rhodot00r_H = (np.matmul(self.H00,self.rho00i)-np.matmul(self.rho00i,self.H00))
        self.rhodot11r_H = (np.matmul(self.H11,self.rho11i)-np.matmul(self.rho11i,self.H11))

    def compute_rhodot_i_H(self):
        self.rhodot00i_H = -1.0 * (np.matmul(self.H00,self.rho00r)-np.matmul(self.rho00r,self.H00))
        self.rhodot11i_H = -1.0 * (np.matmul(self.H11,self.rho11r)-np.matmul(self.rho11r,self.H11))

    def compute_rhodot_r_mu(self,E):
        self.rhodot00r_mu = E * (np.matmul(self.mu01,self.rho10i)-np.matmul(self.rho01i,self.mu10))
        self.rhodot11r_mu = E * (np.matmul(self.mu10,self.rho01i)-np.matmul(self.rho10i,self.mu01))

        self.rhodot01r_mu = E * (np.matmul(self.mu01,self.rho11i)-np.matmul(self.rho00i,self.mu01))
        self.rhodot10r_mu = E * (np.matmul(self.mu10,self.rho00i)-np.matmul(self.rho11i,self.mu10))
        
    def compute_rhodot_i_mu(self,E):
        self.rhodot00i_mu = E * (np.matmul(self.mu01,self.rho10r)-np.matmul(self.rho01r,self.mu10))
        self.rhodot11i_mu = E * (np.matmul(self.mu10,self.rho01r)-np.matmul(self.rho10r,self.mu01))

        self.rhodot01i_mu = E * (np.matmul(self.mu01,self.rho11r)-np.matmul(self.rho00r,self.mu01))
        self.rhodot10i_mu = E * (np.matmul(self.mu10,self.rho00r)-np.matmul(self.rho11r,self.mu10))
        
    def prop_whitenoise_timestep(self):
        compute_rhodot_i_H()
        compute_rhodot_i_mu()
        
    

    

        
