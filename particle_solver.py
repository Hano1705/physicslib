import numpy as np
from wavefunction import Wavefunction
import matplotlib.pyplot as plt

def wave_packet(POSITION_START, MOMENTUM_START, SIGMA):

    wave_function = np.exp(-1j*MOMENTUM_START*X_GRID)*np.exp(-(X_GRID-POSITION_START)**2/(2*SIGMA**2))

    return wave_function

# Grid
DX = 0.01
X_LOWER, X_UPPER = -10, 10
X_GRID = np.arange(X_LOWER, X_UPPER,DX)

# Physical parameters
PLANCKS = 1
MASS = 1
POSITION_START = 0
MOMENTUM_START = 10
SIGMA = 1
POTENTIAL = np.where((X_GRID>-4)&(X_GRID<4),0,100).astype('float64')

wavepacket = Wavefunction(wave_packet(POSITION_START, MOMENTUM_START, SIGMA), X_GRID)
wavepacket.normalize
plt.plot(X_GRID, wavepacket.real)
plt.plot(X_GRID, wavepacket.imag)
plt.show()