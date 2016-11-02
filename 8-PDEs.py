
# coding: utf-8

# In[1]:

import numpy as np
get_ipython().magic('matplotlib inline')
from matplotlib import pyplot as plt


# Adapted from http://kitchingroup.cheme.cmu.edu/pycse/pycse.html#sec-10-4
# 
# ## Plane Poiseuille flow - BVP solve by shooting method
# 
# One approach to solving BVPs is to use the shooting method. The reason we cannot use an initial value solver for a BVP is that there is not enough information at the initial value to start. In the shooting method, we take the function value at the initial point, and guess what the function derivatives are so that we can do an integration. If our guess was good, then the solution will go through the known second boundary point. If not, we guess again, until we get the answer we need. In this example we repeat the pressure driven flow example, but illustrate the shooting method.
# 
# In the pressure driven flow of a fluid with viscosity $\mu$ between two stationary plates separated by distance $d$ and driven by a pressure drop $\Delta P/\Delta x$, the governing equations on the velocity $u$ of the fluid are (assuming flow in the x-direction with the velocity varying only in the y-direction):
# 
# $$\frac{\Delta P}{\Delta x} = \mu \frac{d^2u}{dy^2}$$
# 
# with boundary conditions $u(y=0) = 0$ and $u(y=d) = 0$, i.e. the no-slip condition at the edges of the plate.
# 
# we convert this second order BVP to a system of ODEs by letting $u_1 = u$, $u_2 = u_1'$ and then $u_2' = u_1''$. This leads to:
# 
# $\frac{d u_1}{dy} = u_2$
# 
# $\frac{d u_2}{dy} = \frac{1}{\mu}\frac{\Delta P}{\Delta x}$
# 
# with boundary conditions $u_1(y=0) = 0$ and $u_1(y=d) = 0$.
# 
# for this problem we let the plate separation be d=0.1, the viscosity $\mu = 1$, and $\frac{\Delta P}{\Delta x} = -100$.

# ## Shooting method
# To integrate the differential equations we need starting values $u_1(0)$ and $u_2(0)$, but we only have $u_1(0)=0$. We need to guess a value for $u_2(0)$ and see if the solution goes through the $u_2(d)=0$ boundary value.
# 
# 

# In[2]:

from scipy.integrate import odeint

d = 0.1 # plate thickness
mu = 1
Pdrop = -100

def odefun(U, y):
    u1, u2 = U
    du1dy = u2
    du2dy = 1.0 / mu * Pdrop
    return [du1dy, du2dy]

u1_0 = 0 # known
u2_0 = 1 # guessed

dspan = np.linspace(0, d)

U = odeint(odefun, [u1_0, u2_0], dspan)

def plot_results(dspan, U):
    plt.plot(dspan, U[:,0], label="Shooting method")
    plt.plot([0,d],[0,0], 'ro') # the boundary values as red circles
    plt.xlim(0,d)
    plt.xlabel('d')
    plt.ylabel('$u_1$')

plot_results(dspan, U)


# Too low! Guess higher

# In[3]:

u1_0 = 0 # known
u2_0 = 10 # guessed higher than last time

U = odeint(odefun, [u1_0, u2_0], dspan)

plot_results(dspan, U)


# Rather than keep guessing, let's use a nonlinear solver to converge upon the best value for `u2_0`

# In[4]:

from scipy.optimize import fsolve

u1_0 = 0 # known

def objective(u2_0):
    U = odeint(odefun, [u1_0, u2_0], dspan)
    u1 = U[:,0]
    return u1[-1]

u2_0, = fsolve(objective, 1.0)
print("Optimized u2 is {}".format(u2_0))

# now solve with optimal u2_0
U = odeint(odefun, [u1_0, u2_0], dspan)

plot_results(dspan, U)

# Also plot an analytical solution
u = -(Pdrop) * d**2 / 2 / mu * (dspan / d - (dspan / d)**2)
plt.plot(dspan, u, 'r--', label='Analytical solution')
plt.legend(loc='best')

plt.show()


# ## Finite difference (second order, one dimension)
# Adapted from http://kitchingroup.cheme.cmu.edu/pycse/pycse.html#sec-10-4-2 which was
# adapted from http://www.physics.arizona.edu/~restrepo/475B/Notes/sourcehtml/node24.html
# 
# 
# For this example, we solve the plane poiseuille flow problem using a finite difference approach. An advantage of the approach we use here is we do not have to rewrite the second order ODE as a set of coupled first order ODEs, nor do we have to provide guesses for the solution. We do, however, have to discretize the derivatives and formulate a linear algebra problem.
# 
# 
# > A reminder:
# In the pressure driven flow of a fluid with viscosity $\mu$ between two stationary plates separated by distance $d$ and driven by a pressure drop $\Delta P/\Delta x$, the governing equations on the velocity $u$ of the fluid are (assuming flow in the x-direction with the velocity varying only in the y-direction):
# $$\frac{\Delta P}{\Delta x} = \mu \frac{d^2u}{dy^2}$$
# with boundary conditions $u(y=0) = 0$ and $u(y=d) = 0$, i.e. the no-slip condition at the edges of the plate.
# As before we let the plate separation be $d=0.1$, the viscosity $\mu = 1$, and $\frac{\Delta P}{\Delta x} = -100$.
# 
# 
# The idea behind the finite difference method is to approximate the derivatives by finite differences on a grid. 
# 
# Using central differences
# $$\frac{d^2u}{dx^2} = \frac{1}{\mu} \frac{\Delta P}{\Delta x}$$
# becomes
# $$\frac{u_{i-1} - 2 u_{i} + u_{i+1}}{h^2} = \frac{1}{\mu} \frac{\Delta P}{\Delta x} = -100$$
# 
# 
# By discretizing the ODE, we arrive at a set of linear algebra equations of the form $A y = b$, where $A$ and $b$ are defined as follows.
# 
# $$
# A = \left [ \begin{array}{ccccc} %
#  2         & -1  & 0                    & 0 & 0 \\
# -1  & 2          & -1 & 0 & 0 \\
# 0                    & \ddots               & \ddots               & \ddots & 0 \\
# 0                    & 0                    & -1& 2 & -1  \\
# 0                    & 0                    & 0  & -1  & 2 \end{array} \right ] $$
# 
# 
# $$ y = \left [ \begin{array}{c} y_i \\ \vdots \\ y_N \end{array} \right ] $$
# 
# $$ b = \left [ \begin{array}{c} 100 h^2   \\
#  100 h^2 \\
# \vdots \\
#  100 h^2 \\
#  100 h^2 \end{array} \right] 
# $$
# 
# 
# 
# 
# If we want to solve a **linear boundary value problem** of the **general form**: 
# $$y'' = p(x)y' + q(x)y + r(x)$$
# with boundary conditions $y(x_1) = \alpha$ and $y(x_2) = \beta$.
# then we arrive at a set of linear algebra equations of the form $A y = b$, where $A$ and $b$ are defined as follows.
# 
# $$
# A = \left [ \begin{array}{ccccc} %
#  2 + h^2 q_1         & -1 + \frac{h}{2} p_1 & 0                    & 0 & 0 \\
# -1 - \frac{h}{2} p_2 & 2 + h^2 q_2          & -1 + \frac{h}{2} p_2 & 0 & 0 \\
# 0                    & \ddots               & \ddots               & \ddots & 0 \\
# 0                    & 0                    & -1 - \frac{h}{2} p_{N-1} & 2 + h^2 q_{N-1} & -1 + \frac{h}{2} p_{N-1} \\
# 0                    & 0                    & 0  & -1 - \frac{h}{2} p_N & 2 + h^2 q_N \end{array} \right ] $$
# 
# 
# $$ y = \left [ \begin{array}{c} y_i \\ \vdots \\ y_N \end{array} \right ] $$
# 
# $$ b = \left [ \begin{array}{c} -h^2 r_1 + ( 1 + \frac{h}{2} p_1) \alpha \\
# -h^2 r_2 \\
# \vdots \\
# -h^2 r_{N-1} \\
# -h^2 r_N + (1 - \frac{h}{2} p_N) \beta \end{array} \right] 
# $$
# 
# Mapping our $u'' = \frac{1}{\mu} \frac{\Delta P}{\Delta x}$ onto the general form $$y'' = p(x)y' + q(x)y + r(x)$$ with boundary conditions $y(x_1) = \alpha$ and $y(x_2) = \beta$, we see $p(x)=0$, $q(x)=0$, $r(x)=-100$, $\alpha=0$, $\beta=0$.

# In[5]:

"""
This is set up for general p, q, r, alpha, beta
"""
# we use the notation for y'' = p(x)y' + q(x)y + r(x)
def p(x):
    return 0

def q(x):
    return 0

def r(x):
    return -100

#we use the notation y(x1) = alpha and y(x2) = beta
x1 = 0; alpha = 0.0
x2 = 0.1; beta = 0.0

npoints = 100

# compute interval width
h = (x2-x1)/npoints;

# preallocate and shape the b vector and A-matrix
b = np.zeros((npoints - 1, 1));
A = np.zeros((npoints - 1, npoints - 1));
X = np.zeros((npoints - 1, 1));

#now we populate the A-matrix and b vector elements
for i in range(npoints - 1):
    X[i,0] = x1 + (i + 1) * h

    # get the value of the BVP Odes at this x
    pi = p(X[i])
    qi = q(X[i])
    ri = r(X[i])

    if i == 0:
        # first boundary condition
        b[i] = -h**2 * ri + (1 + h / 2 * pi)*alpha;
    elif i == npoints - 1:
        # second boundary condition
        b[i] = -h**2 * ri + (1 - h / 2 * pi)*beta;
    else:
        b[i] = -h**2 * ri # intermediate points

    for j in range(npoints - 1):
        if j == i: # the diagonal
            A[i,j] = 2 + h**2 * qi
        elif j == i - 1: # left of the diagonal
            A[i,j] = -1 - h / 2 * pi
        elif j == i + 1: # right of the diagonal
            A[i,j] = -1 + h / 2 * pi
        else:
            A[i,j] = 0 # off the tri-diagonal

# solve the equations A*y = b for Y
Y = np.linalg.solve(A,b)

x = np.hstack([x1, X[:,0], x2])
y = np.hstack([alpha, Y[:,0], beta])

plt.plot(x, y, label='finite difference')

mu = 1
d = 0.1
x = np.linspace(0,0.1);
Pdrop = -100 # this is DeltaP/Deltax
u = -(Pdrop) * d**2 / 2.0 / mu * (x / d - (x / d)**2)
plt.plot(x,u,'r--', label='analytical solution')

plt.xlabel('distance between plates')
plt.ylabel('fluid velocity')
plt.legend(loc='best')


# ## Boundary conditions
# This example had a *Dirichlet boundary condition* (fixed value). What if you had a *Neumann boundary condition* (fixed derivative)??
# An example is solving for temperature $T(x)$ you may specify thermal insulation ($dT/dx=0$) at one boundary instead of a fixed temperature.

# 

# In[ ]:



