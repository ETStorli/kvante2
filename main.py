import numpy as np
from numpy import pi, sin, cos, sqrt
import scipy.constants as sc
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import warnings

L = 2*pi  #[nm]
h = L/1e3
v0 = 8
hbar = 1
m = 1
#z = np.arange(L, 1.9*L, h)
z0 = L/(2*hbar) * np.sqrt(2*m*v0)


def funk1(z): return np.tan(z) - np.sqrt(z0**2/z**2 - 1)
def funk2(z): return np.tan(z+pi/2) - np.sqrt(z0**2/z**2 - 1)
def energi(z): return (2*hbar**2*z**2)/(m*L**2) -v0

def plotKrys(f):
    warnings.filterwarnings('ignore')
    z = np.arange(0, 4*pi, .01)
    y = f(z)
    y[:-1][np.diff(y) < 0] = np.nan
    init_gues = [1.4, 4.4, 7.3, 10.3]
    x = fsolve(f, init_gues)
    plt.plot(x, np.zeros(len(x)), 'x', markersize=7)
    plt.plot(z, y)

    intervall = np.arange(0, 4*pi, pi)
    vertikal = [-20, 20]
    for i in intervall:
        plt.plot([i, i], vertikal, '--', color='black', linewidth='.8')
    plt.grid(linestyle='--')
    plt.ylim(-20, 20)


def plotEnerg(energi):
    x = np.array([0., 10.])
    for e in energi:
        plt.plot(x, [e, e])
    plt.xticks([])

init_gues1 = [1.4, 4.4, 7.3, 10.3]
x1 = fsolve(funk1, init_gues1)

init_gues2 = [2.4, 5.4, 8.3, 11.3]
x2 = fsolve(funk2, init_gues2)

e1 = energi(x1)
e2 = energi(x2)

#* Dette er for å dele plotten inn i intervaller med bare ét nullpunkt
x = np.arange(0, 4*pi, pi/2)
y = [-20, 20]

def mkfig_op1():
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plotKrys(funk1)
    plt.subplot(122)
    plotEnerg(e1)
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plotKrys(funk2)
    plt.subplot(122)
    plotEnerg(e2)
    plt.show()


'''
#Vet ikke om dette trengs
def wave_out(x, E):
    a, b = 1, 1
    k = sqrt(2*m*E)/hbar
    return a*sin(k*x) + b*cos(k*x)

def wave_in(x, E):
    c, d = 1, 1
    k = sqrt(2*m*(v0 + E))/hbar
    return c*sin(k*x) + d*cos(k*x)
'''

mkfig_op1()