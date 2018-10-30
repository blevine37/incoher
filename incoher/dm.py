import numpy as np
import math

class dm(object):
    def __init__(self,nk):
        self.rho00 = np.zeros((nk,nk))
        self.rho01 = np.zeros((nk,nk))
        self.rho10 = np.zeros((nk,nk))
        self.rho11 = np.zeros((nk,nk))
        self.nk = nk

    def init_0k(self):
        self.rho00 = np.zeros((self.nk,self.nk))
        self.rho01 = np.zeros((self.nk,self.nk))
        self.rho10 = np.zeros((self.nk,self.nk))
        self.rho11 = np.zeros((self.nk,self.nk))
        self.rho00[0,0] = 1.0

        
