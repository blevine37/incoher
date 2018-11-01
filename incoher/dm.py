import numpy as np
import math
import h5py

# PESs:
# V0 = 1/2 k x^2
# V1 = 1/2 k x^2 + b x + deltae

class dm(object):
    def __init__(self,nk,k,deltae,b,mass,dt,intensity):
        self.nk = nk
        self.nk2 = nk*nk
        self.k = k
        self.deltae = deltae
        self.b = b
        self.mass = mass
        self.omega = math.sqrt(k/mass)
        self.intensity = intensity
        self.dt = dt
        self.time = 0.0

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
        
        self.rhodot00r_H = np.zeros((self.nk,self.nk))
        self.rhodot01r_H = np.zeros((self.nk,self.nk))
        self.rhodot10r_H = np.zeros((self.nk,self.nk))
        self.rhodot11r_H = np.zeros((self.nk,self.nk))
        self.rhodot00i_H = np.zeros((self.nk,self.nk))
        self.rhodot01i_H = np.zeros((self.nk,self.nk))
        self.rhodot10i_H = np.zeros((self.nk,self.nk))
        self.rhodot11i_H = np.zeros((self.nk,self.nk))
        self.rhodot00r_mu = np.zeros((self.nk,self.nk))
        self.rhodot01r_mu = np.zeros((self.nk,self.nk))
        self.rhodot10r_mu = np.zeros((self.nk,self.nk))
        self.rhodot11r_mu = np.zeros((self.nk,self.nk))
        self.rhodot00i_mu = np.zeros((self.nk,self.nk))
        self.rhodot01i_mu = np.zeros((self.nk,self.nk))
        self.rhodot10i_mu = np.zeros((self.nk,self.nk))
        self.rhodot11i_mu = np.zeros((self.nk,self.nk))

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
        
        self.rhodot00r_H = np.zeros((self.nk,self.nk))
        self.rhodot01r_H = np.zeros((self.nk,self.nk))
        self.rhodot10r_H = np.zeros((self.nk,self.nk))
        self.rhodot11r_H = np.zeros((self.nk,self.nk))
        self.rhodot00i_H = np.zeros((self.nk,self.nk))
        self.rhodot01i_H = np.zeros((self.nk,self.nk))
        self.rhodot10i_H = np.zeros((self.nk,self.nk))
        self.rhodot11i_H = np.zeros((self.nk,self.nk))
        self.rhodot00r_mu = np.zeros((self.nk,self.nk))
        self.rhodot01r_mu = np.zeros((self.nk,self.nk))
        self.rhodot10r_mu = np.zeros((self.nk,self.nk))
        self.rhodot11r_mu = np.zeros((self.nk,self.nk))
        self.rhodot00i_mu = np.zeros((self.nk,self.nk))
        self.rhodot01i_mu = np.zeros((self.nk,self.nk))
        self.rhodot10i_mu = np.zeros((self.nk,self.nk))
        self.rhodot11i_mu = np.zeros((self.nk,self.nk))

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

    def build_matrices(self):
        self.build_H()
        self.build_mu()        

    def compute_rhodot_r_H(self):
        self.rhodot00r_H = (np.matmul(self.H00,self.rho00i)-np.matmul(self.rho00i,self.H00))
        self.rhodot11r_H = (np.matmul(self.H11,self.rho11i)-np.matmul(self.rho11i,self.H11))

        self.rhodot01r_H = (np.matmul(self.H00,self.rho01i)-np.matmul(self.rho01i,self.H11))
        self.rhodot10r_H = (np.matmul(self.H11,self.rho10i)-np.matmul(self.rho10i,self.H00))

    def compute_rhodot_i_H(self):
        self.rhodot00i_H = -1.0 * (np.matmul(self.H00,self.rho00r)-np.matmul(self.rho00r,self.H00))
        self.rhodot11i_H = -1.0 * (np.matmul(self.H11,self.rho11r)-np.matmul(self.rho11r,self.H11))

        self.rhodot01i_H = -1.0 * (np.matmul(self.H00,self.rho01r)-np.matmul(self.rho01r,self.H11))
        self.rhodot10i_H = -1.0 * (np.matmul(self.H11,self.rho10r)-np.matmul(self.rho10r,self.H00))

    def compute_rhodot_r_mu(self,E):
        self.rhodot00r_mu = E * (np.matmul(self.mu01,self.rho10i)-np.matmul(self.rho01i,self.mu10))
        self.rhodot11r_mu = E * (np.matmul(self.mu10,self.rho01i)-np.matmul(self.rho10i,self.mu01))

        self.rhodot01r_mu = E * (np.matmul(self.mu01,self.rho11i)-np.matmul(self.rho00i,self.mu01))
        self.rhodot10r_mu = E * (np.matmul(self.mu10,self.rho00i)-np.matmul(self.rho11i,self.mu10))
        
    def compute_rhodot_i_mu(self,E):
        self.rhodot00i_mu = -1.0 * E * (np.matmul(self.mu01,self.rho10r)-np.matmul(self.rho01r,self.mu10))
        self.rhodot11i_mu = -1.0 * E * (np.matmul(self.mu10,self.rho01r)-np.matmul(self.rho10r,self.mu01))

        self.rhodot01i_mu = -1.0 * E * (np.matmul(self.mu01,self.rho11r)-np.matmul(self.rho00r,self.mu01))
        self.rhodot10i_mu = -1.0 * E * (np.matmul(self.mu10,self.rho00r)-np.matmul(self.rho11r,self.mu10))
        
    def prop_whitenoise_timestep(self):

        E = self.intensity * np.random.normal()

        self.compute_rhodot_i_H()
        self.compute_rhodot_i_mu(E)

        rhodot00i = self.rhodot00i_H + self.rhodot00i_mu
        rhodot01i = self.rhodot01i_H + self.rhodot01i_mu
        rhodot10i = self.rhodot10i_H + self.rhodot10i_mu
        rhodot11i = self.rhodot11i_H + self.rhodot11i_mu

        hdt = 0.5 * self.dt

        self.rho00i += hdt * rhodot00i
        self.rho01i += hdt * rhodot01i
        self.rho10i += hdt * rhodot10i
        self.rho11i += hdt * rhodot11i

        self.compute_rhodot_r_H()
        self.compute_rhodot_r_mu(E)

        rhodot00r = self.rhodot00r_H + self.rhodot00r_mu
        rhodot01r = self.rhodot01r_H + self.rhodot01r_mu
        rhodot10r = self.rhodot10r_H + self.rhodot10r_mu
        rhodot11r = self.rhodot11r_H + self.rhodot11r_mu

        self.rho00r += self.dt * rhodot00r
        self.rho01r += self.dt * rhodot01r
        self.rho10r += self.dt * rhodot10r
        self.rho11r += self.dt * rhodot11r

        self.compute_rhodot_i_H()
        self.compute_rhodot_i_mu(E)

        rhodot00i = self.rhodot00i_H + self.rhodot00i_mu
        rhodot01i = self.rhodot01i_H + self.rhodot01i_mu
        rhodot10i = self.rhodot10i_H + self.rhodot10i_mu
        rhodot11i = self.rhodot11i_H + self.rhodot11i_mu

        self.rho00i += hdt * rhodot00i
        self.rho01i += hdt * rhodot01i
        self.rho10i += hdt * rhodot10i
        self.rho11i += hdt * rhodot11i

        self.time += self.dt

    def prop_timestep(self):

        self.compute_rhodot_i_H()

        rhodot00i = self.rhodot00i_H 
        rhodot01i = self.rhodot01i_H 
        rhodot10i = self.rhodot10i_H 
        rhodot11i = self.rhodot11i_H 

        hdt = 0.5 * self.dt

        self.rho00i += hdt * rhodot00i
        self.rho01i += hdt * rhodot01i
        self.rho10i += hdt * rhodot10i
        self.rho11i += hdt * rhodot11i

        self.compute_rhodot_r_H()

        rhodot00r = self.rhodot00r_H 
        rhodot01r = self.rhodot01r_H 
        rhodot10r = self.rhodot10r_H 
        rhodot11r = self.rhodot11r_H 

        self.rho00r += self.dt * rhodot00r
        self.rho01r += self.dt * rhodot01r
        self.rho10r += self.dt * rhodot10r
        self.rho11r += self.dt * rhodot11r

        self.compute_rhodot_i_H()

        rhodot00i = self.rhodot00i_H 
        rhodot01i = self.rhodot01i_H 
        rhodot10i = self.rhodot10i_H 
        rhodot11i = self.rhodot11i_H 

        self.rho00i += hdt * rhodot00i
        self.rho01i += hdt * rhodot01i
        self.rho10i += hdt * rhodot10i
        self.rho11i += hdt * rhodot11i

        self.time += self.dt

    def open_output_file(self,filename):
        self.h5file = h5py.File(filename, "w")
        self.h5file.attrs["nk"] = self.nk
        self.h5file.attrs["naverage"] = 0
        self.h5file.create_dataset("time", (0,1), maxshape=(None,1), dtype="float64")
        self.h5file.create_dataset("rho00r", (0,self.nk2), maxshape=(None,self.nk2), dtype="float64")
        self.h5file.create_dataset("rho00i", (0,self.nk2), maxshape=(None,self.nk2), dtype="float64")
        self.h5file.create_dataset("rho01r", (0,self.nk2), maxshape=(None,self.nk2), dtype="float64")
        self.h5file.create_dataset("rho01i", (0,self.nk2), maxshape=(None,self.nk2), dtype="float64")
        self.h5file.create_dataset("rho10r", (0,self.nk2), maxshape=(None,self.nk2), dtype="float64")
        self.h5file.create_dataset("rho10i", (0,self.nk2), maxshape=(None,self.nk2), dtype="float64")
        self.h5file.create_dataset("rho11r", (0,self.nk2), maxshape=(None,self.nk2), dtype="float64")
        self.h5file.create_dataset("rho11i", (0,self.nk2), maxshape=(None,self.nk2), dtype="float64")

        self.h5file.create_dataset("rhodot00r_H", (0,self.nk2), maxshape=(None,self.nk2), dtype="float64")
        self.h5file.create_dataset("rhodot00i_H", (0,self.nk2), maxshape=(None,self.nk2), dtype="float64")
        self.h5file.create_dataset("rhodot01r_H", (0,self.nk2), maxshape=(None,self.nk2), dtype="float64")
        self.h5file.create_dataset("rhodot01i_H", (0,self.nk2), maxshape=(None,self.nk2), dtype="float64")
        self.h5file.create_dataset("rhodot10r_H", (0,self.nk2), maxshape=(None,self.nk2), dtype="float64")
        self.h5file.create_dataset("rhodot10i_H", (0,self.nk2), maxshape=(None,self.nk2), dtype="float64")
        self.h5file.create_dataset("rhodot11r_H", (0,self.nk2), maxshape=(None,self.nk2), dtype="float64")
        self.h5file.create_dataset("rhodot11i_H", (0,self.nk2), maxshape=(None,self.nk2), dtype="float64")

    def close_output_file(self):
        self.h5file.close()

    def output_to_file(self):
        l = self.h5file["time"].len()
        
        self.h5file["time"].resize(l+1,axis=0)
        self.h5file["rho00r"].resize(l+1,axis=0)
        self.h5file["rho00i"].resize(l+1,axis=0)
        self.h5file["rho01r"].resize(l+1,axis=0)
        self.h5file["rho01i"].resize(l+1,axis=0)
        self.h5file["rho10r"].resize(l+1,axis=0)
        self.h5file["rho10i"].resize(l+1,axis=0)
        self.h5file["rho11r"].resize(l+1,axis=0)
        self.h5file["rho11i"].resize(l+1,axis=0)
        self.h5file["rhodot00r_H"].resize(l+1,axis=0)
        self.h5file["rhodot00i_H"].resize(l+1,axis=0)
        self.h5file["rhodot01r_H"].resize(l+1,axis=0)
        self.h5file["rhodot01i_H"].resize(l+1,axis=0)
        self.h5file["rhodot10r_H"].resize(l+1,axis=0)
        self.h5file["rhodot10i_H"].resize(l+1,axis=0)
        self.h5file["rhodot11r_H"].resize(l+1,axis=0)
        self.h5file["rhodot11i_H"].resize(l+1,axis=0)
        
        self.h5file["time"][l,0] = self.time
        self.h5file["rho00r"][l,:] = self.rho00r.flatten()
        self.h5file["rho00i"][l,:] = self.rho00i.flatten()
        self.h5file["rho01r"][l,:] = self.rho01r.flatten()
        self.h5file["rho01i"][l,:] = self.rho01i.flatten()
        self.h5file["rho10r"][l,:] = self.rho10r.flatten()
        self.h5file["rho10i"][l,:] = self.rho10i.flatten()
        self.h5file["rho11r"][l,:] = self.rho11r.flatten()
        self.h5file["rho11i"][l,:] = self.rho11i.flatten()
        self.h5file["rhodot00r_H"][l,:] = self.rhodot00r_H.flatten()
        self.h5file["rhodot00i_H"][l,:] = self.rhodot00i_H.flatten()
        self.h5file["rhodot01r_H"][l,:] = self.rhodot01r_H.flatten()
        self.h5file["rhodot01i_H"][l,:] = self.rhodot01i_H.flatten()
        self.h5file["rhodot10r_H"][l,:] = self.rhodot10r_H.flatten()
        self.h5file["rhodot10i_H"][l,:] = self.rhodot10i_H.flatten()
        self.h5file["rhodot11r_H"][l,:] = self.rhodot11r_H.flatten()
        self.h5file["rhodot11i_H"][l,:] = self.rhodot11i_H.flatten()

    def prop_whitenoise(self,nsteps,noutput):
        self.build_matrices()
        for istep in range(nsteps):
            if (istep % noutput == 0):
                self.output_to_file()
            self.prop_whitenoise_timestep()
        if (nsteps % noutput == 0):
            self.output_to_file()

    def prop_gaussiankick(self,nsteps,noutput):
        self.build_matrices()
        for istep in range(nsteps):
            if (istep % noutput == 0):
                self.output_to_file()
            if istep == 0:
                self.prop_whitenoise_timestep()
            else:
                self.prop_timestep()
        if (nsteps % noutput == 0):
            self.output_to_file()

    def add_dm_to_average(self,dm):
        n = self.h5file.attrs["naverage"]
        n += 1
        self.h5file.attrs["naverage"] = n
        if n==1:
            l = dm.h5file["time"].len()
            self.h5file["time"].resize(l,axis=0)
            self.h5file["rho00r"].resize(l,axis=0)
            self.h5file["rho00i"].resize(l,axis=0)
            self.h5file["rho01r"].resize(l,axis=0)
            self.h5file["rho01i"].resize(l,axis=0)
            self.h5file["rho10r"].resize(l,axis=0)
            self.h5file["rho10i"].resize(l,axis=0)
            self.h5file["rho11r"].resize(l,axis=0)
            self.h5file["rho11i"].resize(l,axis=0)
            self.h5file["rhodot00r_H"].resize(l,axis=0)
            self.h5file["rhodot00i_H"].resize(l,axis=0)
            self.h5file["rhodot01r_H"].resize(l,axis=0)
            self.h5file["rhodot01i_H"].resize(l,axis=0)
            self.h5file["rhodot10r_H"].resize(l,axis=0)
            self.h5file["rhodot10i_H"].resize(l,axis=0)
            self.h5file["rhodot11r_H"].resize(l,axis=0)
            self.h5file["rhodot11i_H"].resize(l,axis=0)
            
            self.h5file["time"][:,:]   = dm.h5file["time"][:,:]   
            self.h5file["rho00r"][:,:] = dm.h5file["rho00r"][:,:] 
            self.h5file["rho00i"][:,:] = dm.h5file["rho00i"][:,:] 
            self.h5file["rho01r"][:,:] = dm.h5file["rho01r"][:,:] 
            self.h5file["rho01i"][:,:] = dm.h5file["rho01i"][:,:] 
            self.h5file["rho10r"][:,:] = dm.h5file["rho10r"][:,:] 
            self.h5file["rho10i"][:,:] = dm.h5file["rho10i"][:,:] 
            self.h5file["rho11r"][:,:] = dm.h5file["rho11r"][:,:] 
            self.h5file["rho11i"][:,:] = dm.h5file["rho11i"][:,:]
            self.h5file["rhodot00r_H"][:,:] = dm.h5file["rhodot00r_H"][:,:] 
            self.h5file["rhodot00i_H"][:,:] = dm.h5file["rhodot00i_H"][:,:] 
            self.h5file["rhodot01r_H"][:,:] = dm.h5file["rhodot01r_H"][:,:] 
            self.h5file["rhodot01i_H"][:,:] = dm.h5file["rhodot01i_H"][:,:] 
            self.h5file["rhodot10r_H"][:,:] = dm.h5file["rhodot10r_H"][:,:] 
            self.h5file["rhodot10i_H"][:,:] = dm.h5file["rhodot10i_H"][:,:] 
            self.h5file["rhodot11r_H"][:,:] = dm.h5file["rhodot11r_H"][:,:] 
            self.h5file["rhodot11i_H"][:,:] = dm.h5file["rhodot11i_H"][:,:]
        else:
            x = 1.0 / n
            y = 1.0 - x
            self.h5file["rho00r"][:,:] = y * self.h5file["rho00r"][:,:] + x * dm.h5file["rho00r"][:,:]
            self.h5file["rho00i"][:,:] = y * self.h5file["rho00i"][:,:] + x * dm.h5file["rho00i"][:,:]
            self.h5file["rho01r"][:,:] = y * self.h5file["rho01r"][:,:] + x * dm.h5file["rho01r"][:,:]
            self.h5file["rho01i"][:,:] = y * self.h5file["rho01i"][:,:] + x * dm.h5file["rho01i"][:,:]
            self.h5file["rho10r"][:,:] = y * self.h5file["rho10r"][:,:] + x * dm.h5file["rho10r"][:,:]
            self.h5file["rho10i"][:,:] = y * self.h5file["rho10i"][:,:] + x * dm.h5file["rho10i"][:,:]
            self.h5file["rho11r"][:,:] = y * self.h5file["rho11r"][:,:] + x * dm.h5file["rho11r"][:,:]
            self.h5file["rho11i"][:,:] = y * self.h5file["rho11i"][:,:] + x * dm.h5file["rho11i"][:,:]
            self.h5file["rhodot00r_H"][:,:] = y * self.h5file["rhodot00r_H"][:,:] + x * dm.h5file["rhodot00r_H"][:,:]
            self.h5file["rhodot00i_H"][:,:] = y * self.h5file["rhodot00i_H"][:,:] + x * dm.h5file["rhodot00i_H"][:,:]
            self.h5file["rhodot01r_H"][:,:] = y * self.h5file["rhodot01r_H"][:,:] + x * dm.h5file["rhodot01r_H"][:,:]
            self.h5file["rhodot01i_H"][:,:] = y * self.h5file["rhodot01i_H"][:,:] + x * dm.h5file["rhodot01i_H"][:,:]
            self.h5file["rhodot10r_H"][:,:] = y * self.h5file["rhodot10r_H"][:,:] + x * dm.h5file["rhodot10r_H"][:,:]
            self.h5file["rhodot10i_H"][:,:] = y * self.h5file["rhodot10i_H"][:,:] + x * dm.h5file["rhodot10i_H"][:,:]
            self.h5file["rhodot11r_H"][:,:] = y * self.h5file["rhodot11r_H"][:,:] + x * dm.h5file["rhodot11r_H"][:,:]
            self.h5file["rhodot11i_H"][:,:] = y * self.h5file["rhodot11i_H"][:,:] + x * dm.h5file["rhodot11i_H"][:,:]
            
    def smear_in_time(self,dm):
        l = dm.h5file["time"].len()
        self.h5file["time"].resize(l,axis=0)
        self.h5file["rho00r"].resize(l,axis=0)
        self.h5file["rho00i"].resize(l,axis=0)
        self.h5file["rho01r"].resize(l,axis=0)
        self.h5file["rho01i"].resize(l,axis=0)
        self.h5file["rho10r"].resize(l,axis=0)
        self.h5file["rho10i"].resize(l,axis=0)
        self.h5file["rho11r"].resize(l,axis=0)
        self.h5file["rho11i"].resize(l,axis=0)
        self.h5file["rhodot00r_H"].resize(l,axis=0)
        self.h5file["rhodot00i_H"].resize(l,axis=0)
        self.h5file["rhodot01r_H"].resize(l,axis=0)
        self.h5file["rhodot01i_H"].resize(l,axis=0)
        self.h5file["rhodot10r_H"].resize(l,axis=0)
        self.h5file["rhodot10i_H"].resize(l,axis=0)
        self.h5file["rhodot11r_H"].resize(l,axis=0)
        self.h5file["rhodot11i_H"].resize(l,axis=0)
        
        self.h5file["time"][:,:]   = dm.h5file["time"][:,:]   
        self.h5file["rho00r"][:,:] = dm.h5file["rho00r"][:,:] 
        self.h5file["rho00i"][:,:] = dm.h5file["rho00i"][:,:] 
        self.h5file["rho01r"][:,:] = dm.h5file["rho01r"][:,:] 
        self.h5file["rho01i"][:,:] = dm.h5file["rho01i"][:,:] 
        self.h5file["rho10r"][:,:] = dm.h5file["rho10r"][:,:] 
        self.h5file["rho10i"][:,:] = dm.h5file["rho10i"][:,:] 
        self.h5file["rho11r"][:,:] = dm.h5file["rho11r"][:,:] 
        self.h5file["rho11i"][:,:] = dm.h5file["rho11i"][:,:]
        self.h5file["rhodot00r_H"][:,:] = dm.h5file["rhodot00r_H"][:,:] 
        self.h5file["rhodot00i_H"][:,:] = dm.h5file["rhodot00i_H"][:,:] 
        self.h5file["rhodot01r_H"][:,:] = dm.h5file["rhodot01r_H"][:,:] 
        self.h5file["rhodot01i_H"][:,:] = dm.h5file["rhodot01i_H"][:,:] 
        self.h5file["rhodot10r_H"][:,:] = dm.h5file["rhodot10r_H"][:,:] 
        self.h5file["rhodot10i_H"][:,:] = dm.h5file["rhodot10i_H"][:,:] 
        self.h5file["rhodot11r_H"][:,:] = dm.h5file["rhodot11r_H"][:,:] 
        self.h5file["rhodot11i_H"][:,:] = dm.h5file["rhodot11i_H"][:,:]

        rho00r_t0 = dm.h5file["rho00r"][0,:]
        rho00i_t0 = dm.h5file["rho00i"][0,:]
        rho01r_t0 = dm.h5file["rho01r"][0,:]
        rho01i_t0 = dm.h5file["rho01i"][0,:]
        rho10r_t0 = dm.h5file["rho10r"][0,:]
        rho10i_t0 = dm.h5file["rho10i"][0,:]
        rho11r_t0 = dm.h5file["rho11r"][0,:]
        rho11i_t0 = dm.h5file["rho11i"][0,:]
        rhodot00r_t0 = dm.h5file["rhodot00r_H"][0,:]
        rhodot00i_t0 = dm.h5file["rhodot00i_H"][0,:]
        rhodot01r_t0 = dm.h5file["rhodot01r_H"][0,:]
        rhodot01i_t0 = dm.h5file["rhodot01i_H"][0,:]
        rhodot10r_t0 = dm.h5file["rhodot10r_H"][0,:]
        rhodot10i_t0 = dm.h5file["rhodot10i_H"][0,:]
        rhodot11r_t0 = dm.h5file["rhodot11r_H"][0,:]
        rhodot11i_t0 = dm.h5file["rhodot11i_H"][0,:]
        
        for itime in range(1,l):
            rho00r_t = dm.h5file["rho00r"][itime,:]
            rho00i_t = dm.h5file["rho00i"][itime,:]
            rho01r_t = dm.h5file["rho01r"][itime,:]
            rho01i_t = dm.h5file["rho01i"][itime,:]
            rho10r_t = dm.h5file["rho10r"][itime,:]
            rho10i_t = dm.h5file["rho10i"][itime,:]
            rho11r_t = dm.h5file["rho11r"][itime,:]
            rho11i_t = dm.h5file["rho11i"][itime,:]
            rhodot00r_t = dm.h5file["rhodot00r_H"][itime,:]
            rhodot00i_t = dm.h5file["rhodot00i_H"][itime,:]
            rhodot01r_t = dm.h5file["rhodot01r_H"][itime,:]
            rhodot01i_t = dm.h5file["rhodot01i_H"][itime,:]
            rhodot10r_t = dm.h5file["rhodot10r_H"][itime,:]
            rhodot10i_t = dm.h5file["rhodot10i_H"][itime,:]
            rhodot11r_t = dm.h5file["rhodot11r_H"][itime,:]
            rhodot11i_t = dm.h5file["rhodot11i_H"][itime,:]

            rho00r_diff = rho00r_t - rho00r_t0
            rho00i_diff = rho00i_t - rho00i_t0
            rho01r_diff = rho01r_t - rho01r_t0
            rho01i_diff = rho01i_t - rho01i_t0
            rho10r_diff = rho10r_t - rho10r_t0
            rho10i_diff = rho10i_t - rho10i_t0
            rho11r_diff = rho11r_t - rho11r_t0
            rho11i_diff = rho11i_t - rho11i_t0
            rhodot00r_diff = rhodot00r_t - rhodot00r_t0
            rhodot00i_diff = rhodot00i_t - rhodot00i_t0
            rhodot01r_diff = rhodot01r_t - rhodot01r_t0
            rhodot01i_diff = rhodot01i_t - rhodot01i_t0
            rhodot10r_diff = rhodot10r_t - rhodot10r_t0
            rhodot10i_diff = rhodot10i_t - rhodot10i_t0
            rhodot11r_diff = rhodot11r_t - rhodot11r_t0
            rhodot11i_diff = rhodot11i_t - rhodot11i_t0

            for jtime in range(itime+1,l):
                self.h5file["rho00r"][jtime,:] += rho00r_diff
                self.h5file["rho00i"][jtime,:] += rho00i_diff
                self.h5file["rho01r"][jtime,:] += rho01r_diff
                self.h5file["rho01i"][jtime,:] += rho01i_diff
                self.h5file["rho10r"][jtime,:] += rho10r_diff
                self.h5file["rho10i"][jtime,:] += rho10i_diff
                self.h5file["rho11r"][jtime,:] += rho11r_diff
                self.h5file["rho11i"][jtime,:] += rho11i_diff
                self.h5file["rhodot00r_H"][jtime,:] += rhodot00r_diff
                self.h5file["rhodot00i_H"][jtime,:] += rhodot00i_diff
                self.h5file["rhodot01r_H"][jtime,:] += rhodot01r_diff
                self.h5file["rhodot01i_H"][jtime,:] += rhodot01i_diff
                self.h5file["rhodot10r_H"][jtime,:] += rhodot10r_diff
                self.h5file["rhodot10i_H"][jtime,:] += rhodot10i_diff
                self.h5file["rhodot11r_H"][jtime,:] += rhodot11r_diff
                self.h5file["rhodot11i_H"][jtime,:] += rhodot11i_diff
                
  
        

    

        
