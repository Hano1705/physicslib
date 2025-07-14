import numpy as np
from hamiltonian import Hamiltonian
from parameters import GRID, DT, POTENTIAL, UPPER, LOWER
from wavefunction import Wavefunction, wave_packet
import matplotlib.pyplot as plt
from matplotlib import animation

psi0 = Wavefunction(wave_packet())

psi = []
psi.append(psi0.wavefunction)

runtime = 5


def time_evolver():
    
    H = Hamiltonian()

    forward_unitary = np.eye(GRID.size) - 1j * DT/2 * H.hamiltonian
    backward_unitary = np.eye(GRID.size) + 1j * DT/2 * H.hamiltonian

    t = 0
    while t < runtime:
        t += DT
        print(t)
        b = np.empty(psi0.wavefunction.shape, dtype='complex64')
        b = np.matmul(forward_unitary, psi[-1], out=b)
        new_psi = np.linalg.solve(backward_unitary, b)
        psi.append(new_psi)

time_evolver()

def animate(frame):
    
    prob.set_data((GRID, np.abs(psi[frame])**2))
    re.set_data((GRID, np.real(psi[frame])))
    
    return prob,re

fig, ax = plt.subplots()

ax.set_xlim((LOWER-1, UPPER+1))
ax.set_ylim((-3, 3))

prob, = ax.plot([], [])
re, = ax.plot([], [], '--')
ax.set_title(f'Time: 0')

def box_init():
    plt.gcf().axes[0].axvspan(UPPER, UPPER+1, alpha=0.2, color='red')
    plt.gcf().axes[0].axvspan(LOWER-1, LOWER, alpha=0.2, color='red')
    plt.xlim(LOWER-1,UPPER+1)
    plt.ylim((-2*np.max(np.abs(psi)), 2*np.max(np.abs(psi))))

anim = animation.FuncAnimation(fig, animate, frames=round(runtime/DT), interval=200, init_func=box_init)


FFwriter = animation.FFMpegWriter(fps=60)
anim.save('animation.mp4', writer=FFwriter)