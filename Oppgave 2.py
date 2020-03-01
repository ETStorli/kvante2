import numpy as np
from numpy import pi, sin, cos, sqrt
import scipy.constants as sc
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

hbar = 1
m = 1
Dx = .01
w = 1
L = pi
v0 = 8

v = lambda x: -v0 if (abs(x-L/2)<L/2) else 0

def potensial(x): return 0.5*m*w**2*x**2

#n = odde = eksisterte tilstand nr. n
def psi_odde(E, V, L):
    x = np.arange(0, L, Dx)
    psiH = np.zeros(len(x))
    psiH[0] = 0
    psiH[1] = 1
    for i in range(len(x)-2):
        psiH[i+2] = 2*m*Dx**2/hbar**2*(V(x[i+1]) - E)*psiH[i+1] + 2*psiH[i+1] - psiH[i]
    #Pga. symmetri, og for å ikke ta med 'psi0' to ganger:
    psiV = -psiH[-1:0:-1]
    psi = np.append(psiV, psiH)
    x = np.append(-x[-1:0:-1], x)
    return x, psi


def psi_jevn(E, V, L):
    x = np.arange(0, L, Dx, dtype=np.float64)
    psiH = np.zeros(len(x))
    psiH[0] = 1
    psiH[1] = (m*Dx**2/hbar**2 *(V(x[0]) - E) + 1)*psiH[0]
    for i in range(len(x)-2):
        psiH[i+2] = 2*m*Dx**2/hbar**2*(V(x[i+1]- E))*psiH[i+1] + 2*psiH[i+1] - psiH[i] 
    #    print(psiH[i+1])
    #Pga. symmetri, og for å ikke ta med 'psi0' to ganger:
    psiV = psiH[-1:0:-1]
    psi = np.append(psiV, psiH)
    x = np.append(-x[-1:0:-1], x)
    return x, psi

'''
for i in np.arange(-10, 10, .1):
    x_j, psi_j = psi_jevn(i, potensial, 2*np.pi)
    print(max(psi_j), end=": ")
    print(i)
'''

x_j, psi_j = psi_jevn(-7, v, 10*np.pi)
x_o, psi_o = psi_odde(-10, v, 10*np.pi)

#plt.plot(x_j, psi_j)
plt.plot(x_o, psi_o)
#t = int(len(psi_j)/2)

#plt.show()
#plt.plot(x, potensial(x))
plt.show()

