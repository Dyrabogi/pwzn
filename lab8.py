import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


t = sp.symbols('t', real=True)
m, c, k, F0, omega = sp.symbols('m c k F0 omega', positive=True)
#masa, c(tłumienie),k, amplituda ..
C1, C2 = sp.symbols('C1 C2')
x = sp.Function('x')

oscDamp = sp.Eq(
    m*x(t).diff(t, 2) + c*x(t).diff(t) + k*x(t),
    0
)
dampSolution = sp.dsolve(oscDamp).rhs


oscForced = sp.Eq(
    m*x(t).diff(t, 2) + k*x(t),
    F0*sp.cos(omega*t)
)
forcedSolution = sp.dsolve(oscForced).rhs


oscComb = sp.Eq(
    m*x(t).diff(t, 2) + c*x(t).diff(t) + k*x(t),
    F0*sp.cos(omega*t)
)
combSolution = sp.dsolve(oscComb).rhs


params = {m: 1.0, c: 0.4, k: 4.0, F0: 1.0, omega: 1.3, C1: 1.0, C2: 0.0 }


xDamped = sp.lambdify(t, dampSolution.subs(params), "numpy")
xForced = sp.lambdify(t, forcedSolution.subs(params), "numpy")
xComb   = sp.lambdify(t, combSolution.subs(params), "numpy")


tAxis = np.linspace(0, 30, 2000)


plt.figure(figsize=(4, 3),dpi=301)
plt.plot(tAxis, xDamped(tAxis), label="tłumiony")
plt.xlabel("t")
plt.ylabel("x(t)")
plt.grid(True)
plt.show()

plt.figure(figsize=(4,3),dpi=302)
plt.plot(tAxis, xForced(tAxis), label="wymuszony")
plt.xlabel("t")
plt.ylabel("x(t)")
plt.grid(True)
plt.show()



plt.figure(figsize=(4,3),dpi=303)
plt.plot(tAxis, xComb(tAxis), label="tłumiony + wymuszony")
plt.xlabel("t")
plt.ylabel("x(t)")
plt.grid(True)
plt.show()

plt.figure(figsize=(11,7))
plt.plot(tAxis, xDamped(tAxis), label="tłumiony")
plt.plot(tAxis, xForced(tAxis), label="wymuszony")
plt.plot(tAxis, xComb(tAxis), label="tłumiony + wymuszony")
plt.xlabel("t")
plt.ylabel("x(t)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()