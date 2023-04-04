# Advect - program to solve the advection equation using the various hyperbolic PDE schemes.

# Set up configuration options and special features
import numpy as np
import matplotlib.pyplot as plt


# Select the numerical parameters (time step, grid spacing, etc.)
method = eval(input('Choose a numerical method: 1) FTCS; 2) Lax; 3)Lax-Wendroff: '))
N = eval(input('Enter number of grid points: '))
L = 1.0  # System size
h = L/N  # Grid spacing
c = 1.0  # Wave speed

print('Time for wave to move one grid spacing is {0:.2f}'.format(h/c))

tau = eval(input('Enter time step: '))
coeff = -c*tau/(2.0*h)  # Coefficient used by all schemes
coefflw = 2*coeff**2  # Coefficient used by L-W scheme

print('Wave circles the system in {0:.2f} steps'.format(L/(c*tau)))
nStep = eval(input('Enter total number of steps: '))

# Set initial and boundary conditions
sigma = 0.1  # Width of the Gaussian pulses
k_wave = np.pi/sigma  # Wave number of the cosine
x = np.arange(N)*h - L/2  # Coordinates of the grid points

# Set up initial condition to be a Gaussian-cosine pulse
a = np.empty(N)
for i in range(N):
    a[i] = np.cos(k_wave*x[i]) * np.exp(-x[i]**2/(2*sigma**2))

# Use periodic boundary conditions
ip = np.arange(N) + 1
ip[N-1] = 0  # ip  = i+1 with periodic b.c.
im = np.arange(N) - 1
im[0] = N-1  # im = i-1 with periodic b.c.

# Initialize plotting variables
iplot = 1  # Plot counter
nplots = 50  # Desired number of plots
aplot = np.empty((N, nplots))
tplot = np.empty(nplots)