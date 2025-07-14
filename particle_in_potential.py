from scipy.integrate import RK45,RK23
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
'''
NOTES:  1) at the moment code is very messy.
        2) DX smaller than 0.1 makes RK45 entirely too slow, and at this resolution, the wavefunction wiggles in an odd manner.
        3) One period is 2*PI/E_n since hbar = 1

'''

def wave_packet(POSITION_START, MOMENTUM_START, SIGMA):
    wave_function = np.exp(-1j*MOMENTUM_START*X_GRID)*np.exp(-(X_GRID-POSITION_START)**2/(2*SIGMA**2))
    norm = np.sum(np.abs(wave_function)**2*DX)
    return wave_function/np.sqrt(norm)

def box_eigenstate(n=1):
    wave_function = np.sin(n*np.pi/LENGTH*(X_GRID+LENGTH/2)).astype('complex64')
    norm = np.sum(np.abs(wave_function)**2*DX)
    return wave_function/np.sqrt(norm)

def d_dxdx(psi):
    dphi_dxdx = -2*psi
    dphi_dxdx[1:] += psi[:-1]
    dphi_dxdx[:-1] += psi[1:]
    return dphi_dxdx/DX**2 # checked for cos, must be squared

def d_dt(t, psi):
    hamiltonian_psi = -PLANCKS**2/(2*MASS)*d_dxdx(psi) + POTENTIAL*psi
    #hamiltonian_psi += -1j*(hamiltonian_psi-np.pi**2/(2*MASS)/LENGTH**2*psi)
    return -1j/PLANCKS*hamiltonian_psi

def std(x,y):
    m = (x * y).sum() / y.sum()
    return np.sqrt(np.sum((x - m)**2 * y) / y.sum())

# def eigenvalue(psi):
#     hamiltonian_psi = -PLANCKS**2/(2*MASS)*d_dxdx(np.real(psi)) + POTENTIAL*np.real(psi)
#     eigenvalue = np.divide(hamiltonian_psi,np.real(psi))
#     return eigenvalue

# Grid
X_LOWER, X_UPPER = -10, 10
X_GRID = np.linspace(X_LOWER, X_UPPER, 1000)
DX = X_GRID[1]-X_GRID[0]
LENGTH = X_UPPER - X_LOWER

# Physical parameters
PLANCKS = 1
MASS = 0.5
POSITION_START = 4
MOMENTUM_START = 10
SIGMA = 0.5
BARRIER_DISTANCE = 5*np.pi/MOMENTUM_START
POTENTIAL = np.where((X_GRID>-0.05)&(X_GRID<0.05),200,0).astype('float64')
POTENTIAL += np.where((X_GRID>-0.05-BARRIER_DISTANCE)
                      &(X_GRID<0.05-BARRIER_DISTANCE),200,0).astype('float64')
#POTENTIAL += 10*X_GRID**2/np.max(X_GRID**2)
# b = np.pi/k 

DT_DXDX = 0.05
t0 = 0
psi0 = wave_packet(POSITION_START, MOMENTUM_START, SIGMA)

solver = RK45(d_dt, t0, psi0, t_bound=2, max_step=DT_DXDX*np.square(DX))

time = []
psi = []

time.append(0)
psi.append(psi0)
i=0
while True:
    i+=1
    solver.step()     
    if i%round(1/(DT_DXDX*np.square(DX))/1000) == 0:
        print(i)
        print(solver.t)   
        time.append(solver.t)
        wave_step = solver.y
        norm = np.sum(np.abs(wave_step)**2*DX)
        wave_step = wave_step/np.sqrt(norm)
        psi.append(wave_step)
    
    if solver.status == 'finished':
        break

fig, ax = plt.subplots()

ax.set_xlim((X_LOWER-1, X_UPPER+1))
ax.set_ylim((-3, 3))

prob, = ax.plot([], [])
re, = ax.plot([], [], '--')
ax.set_title(f'Time: 0')

def box_init():
    plt.gcf().axes[0].axvspan(X_UPPER, X_UPPER+1, alpha=0.2, color='red')
    plt.gcf().axes[0].axvspan(X_LOWER-1, X_LOWER, alpha=0.2, color='red')
    plt.fill_between(X_GRID,POTENTIAL/np.max(POTENTIAL),color='blue',alpha=0.2)
    plt.xlim(X_LOWER-1,X_UPPER+1)
    plt.ylim((-2*np.max(np.abs(psi)), 2*np.max(np.abs(psi))))


def animate(frame):
    
    prob.set_data((X_GRID, np.abs(psi[frame])**2))
    re.set_data((X_GRID, np.real(psi[frame])))
    time_passed = round(time[frame],0)
    ax.set_title(f'Time:{time_passed}')
    
    return prob,re


anim = animation.FuncAnimation(fig, animate, frames=len(time), interval=200, init_func=box_init)


FFwriter = animation.FFMpegWriter(fps=60)
anim.save('animation.mp4', writer=FFwriter)