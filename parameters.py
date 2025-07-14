import numpy as np

# Grid
LOWER, UPPER = -10, 10
GRID = np.linspace(LOWER, UPPER, 1000)
DX = GRID[1]-GRID[0]
LENGTH = UPPER - LOWER

# Time evolution
DT = 0.001

# Physical parameters
HBAR = 1
MASS = 1

# Potential
POTENTIAL = np.where((GRID>-UPPER) & (GRID<LOWER),0,100).astype('float64')

# Initial wavefunction
POSITION_START = 0
MOMENTUM_START = 10
SIGMA = 0.5

