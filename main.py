import numpy as np
from numpy import pi
import scipy.constants as sc
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

L = 2*pi  #[nm]
h = L/1e3
v0 = 8
hbar = 1
m = 1
#z = np.arange(L, 1.9*L, h)
z0 = L/(2*hbar) * np.sqrt(2*m*v0)


def funk1(z): return np.tan(z) - np.sqrt(z0**2/z**2 - 1)
def funk2(z): return np.tan(z+pi/2) - np.sqrt(z0**2/z**2 - 1)

def plotF1():
    z = np.arange(0, 4*pi, .01)
    y = funk1(z)
    y[:-1][np.diff(y) < 0] = np.nan
    init_gues = [1.4, 4.4, 7.3, 10.3]
    x = fsolve(funk1, init_gues)
    print(x)
    plt.plot(z, y, label='f1')
    plt.plot(x, np.zeros(len(x)), 'x')
    plt.grid(linestyle='--')
    plt.ylim(-20, 20)

def plotF2():
    z = np.arange(0, 4*pi, .01)
    y = funk2(z)
    y[:-1][np.diff(y) < 0] = np.nan
    init_gues = [2.4, 5.4, 8.3, 11.3]
    x = fsolve(funk2, init_gues)
    plt.plot(z, y, label='f2')
    plt.plot(x, np.zeros(len(x)), 'x')
    plt.grid(linestyle='--')
    plt.ylim(-20, 20)
    plt.legend()

#* Dette er for å dele plotten inn i intervaller med bare ét nullpunkt
x = np.arange(0, 4*pi, pi/2)
y = [-20, 20]

for i in x:
    plt.plot([i, i], y, '--', color='black', linewidth='.8')
plotF1()
plotF2()
plt.show()
