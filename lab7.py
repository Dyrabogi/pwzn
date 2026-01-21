import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import numba

S0 = 999  # Initial susceptible population
I0 = 1     # Initial infected population
R0 = 0     # Initial recovered population
beta = 0.3  # Infection rate
gamma = 0.2  # Recovery rate

beta2 = 0.6  
gamma2 = 0.1  

beta3 = 0.2 
gamma3 = 0.3  

beta4 = 0.4  
gamma4 = 0.4  

N = S0 + I0 + R0

@numba.njit
def sirModel(y, t, beta, gamma, N):
    S, I, R = y
    dSdT = -beta/N * I * S
    dIdT = (beta/N * I * S) - (gamma*I)
    dRdT = gamma*I
    return [dSdT, dIdT, dRdT]



t = np.linspace(0,200,100)

y0 = [S0, I0, R0]

solution = odeint(sirModel, y0, t, args=(beta, gamma, N))
solution2 = odeint(sirModel, y0, t, args=(beta2, gamma2, N))
solution3 = odeint(sirModel, y0, t, args=(beta3, gamma3, N))
solution4 = odeint(sirModel, y0, t, args=(beta4, gamma4, N))

S, I, R = solution.T
S2, I2, R2 = solution2.T
S3, I3, R3 = solution3.T
S4, I4, R4 = solution4.T

plt.figure(figsize=(4, 3), dpi=300)
plt.plot(t, S, lw=2, label='$S(t)$')
plt.plot(t, I, lw=2, label='$I(t)$')
plt.plot(t, R, lw=2, label='$R(t)$')
plt.title('beta=0.3, gamma=0.2')
plt.legend()
plt.show()
plt.figure(figsize=(4, 3), dpi=301)
plt.plot(t, S2, lw=2, label='$S_2(t)$')
plt.plot(t, I2, lw=2, label='$I_2(t)$')
plt.plot(t, R2, lw=2, label='$R_2(t)$')
plt.title('beta=0.6, gamma=0.1')
plt.legend()
plt.show()
plt.figure(figsize=(4, 3), dpi=302)
plt.plot(t, S3, lw=2, label='$S_3(t)$')
plt.plot(t, I3, lw=2, label='$I_3(t)$')
plt.plot(t, R3, lw=2, label='$R_3(t)$')
plt.title('beta=0.2, gamma=0.3')
plt.legend()
plt.show()
plt.figure(figsize=(4, 3), dpi=303)
plt.plot(t, S4, lw=2, label='$S_4(t)$')
plt.plot(t, I4, lw=2, label='$I_4(t)$')
plt.plot(t, R4, lw=2, label='$R_4(t)$')
plt.title('beta=0.4, gamma=0.4')
plt.legend()
plt.show()
plt.savefig('logistic_growth_model.png')
plt.savefig('logistic_growth_model.pdf')