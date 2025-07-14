import numpy as np
import scipy.sparse as spa
import matplotlib.pyplot as plt
from parameters import GRID, HBAR, MASS, POTENTIAL


class Hamiltonian():

    def __init__(self):

        ones_array = np.ones(GRID.size)

        self.kinetic = -HBAR**2/2/MASS/(GRID[1]-GRID[0])**2 * spa.spdiags(
            [ones_array, -2*ones_array, ones_array], [-1, 0, 1], GRID.size, GRID.size)
        
        self.potential = spa.spdiags([POTENTIAL],[0], GRID.size, GRID.size)

        self.hamiltonian = self.kinetic + self.potential

    def diagonalize(self):

        return np.linalg.eigh(self.hamiltonian.toarray())

my_matrix = Hamiltonian()
e, v = my_matrix.diagonalize()

