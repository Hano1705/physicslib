import numpy as np
from parameters import GRID, DX, LENGTH, POSITION_START, MOMENTUM_START, SIGMA

class Wavefunction():

    def __init__(self, function):

        self.wavefunction = function.astype(np.complex64)
        self.norm = np.sqrt(np.sum(np.abs(function)**2 * DX))
        self.real = np.real(self.wavefunction)
        self.imag = np.imag(self.wavefunction)
        self.abs = np.abs(self.wavefunction)
        self.prob = self.abs**2

    def normalize(self):

        self.wavefunction = self.wavefunction / self.norm

def wave_packet():

    wave_function = np.exp(-1j * MOMENTUM_START * GRID) * np.exp(-(GRID-POSITION_START)**2 / (2 * SIGMA**2))
    
    norm = np.sum(np.abs(wave_function)**2*DX)
    
    return wave_function/np.sqrt(norm)

def box_eigenstate(n=1):
    wave_function = np.sin(n*np.pi/LENGTH*(GRID+LENGTH/2)).astype('complex64')

    norm = np.sum(np.abs(wave_function)**2*DX)
    
    return wave_function/np.sqrt(norm)