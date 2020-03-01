import numpy as np
from numpy import pi, sin, cos, sqrt
import scipy.constants as sc
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

hbar = 1
m = 1
Dx = .01
w = 1
v0 = 8
L = 2*pi

v = lambda x: -v0 if (x<0<L) else 0
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
    psiH[1] = (m*Dx**2/hbar**2 * (V(x[0]) - E) +1)*psiH[0]
    for i in range(len(x)-2):
        psiH[i+2] = 2*m*Dx**2/hbar**2 * (V(x[i+1]) - E)*psiH[i+1] + 2*psiH[i+1] - psiH[i]
    #    print(psiH[i+1])
    #Pga. symmetri, og for å ikke ta med 'psi0' to ganger:
    psiV = psiH[-1:0:-1]
    psi = np.append(psiV, psiH)
    x = np.append(-x[-1:0:-1], x)
    return x, psi

x, psi = psi_jevn(.49*w*hbar, v, 2*np.pi)
x1, psi1 = psi_jevn(.51*w*hbar, v, 2*np.pi)
xo, psio = psi_odde(.49*w*hbar, v, 2*np.pi)
x1o, psi1o = psi_odde(.51*w*hbar, v, 2*np.pi)

print(psi)
plt.plot([x[0], x[-1]], [0,0], linestyle='--', linewidth=.8, color='black') # Lager en line y=0
plt.plot(x, psi)
plt.plot(x1, psi1, 'g')

#plt.show()
#plt.plot(x, potensial(x))
plt.show()


plt.plot(xo, psio)
plt.plot([x[0], x[-1]], [0, 0], linestyle='--', linewidth=.8, color='black') # Lager en line y=0
plt.plot(x1o, psi1o, 'g')
plt.show()
