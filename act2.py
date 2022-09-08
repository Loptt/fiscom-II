import math
import numpy as np
import matplotlib.pyplot as plt

N = 40                        # Número de puntos
L = 0.5                       # Longitud del espacio
tao = 0.001                   # Paso en tiempo
tf = (N-1)*tao                # Tiempo final
kappa = 0.08                  # Coeficiente de difusión
t = np.linspace(0,tf,N)          # Vector de Tiempo
h = L/(N-1)                   # Paso en espacio
x = np.linspace(-L/2,L/2,N)      # Vector distancia
alpha = kappa*tao/(h**2)         # Parámetro de propagación

## Condición Inicial

# Delta de Dirac

# T1 = ((x+h)/h^2).(-h<x).(x<0);
# T2 = (x-h)/h^2.(0<x).(x<h);
# T0 = zeros(1,N) +T1 +T2;


# Gaussiana

sigma = 0.1
x0 = 0
T0 = (1 / (sigma * math.sqrt(2*math.pi))) * np.exp(-1 * np.square(x - x0) / (2*(sigma**2)))

## Propagacion

diag_prin = np.ones(N) * (1 - 2*alpha)
diag_alphas = np.ones(N-1) * alpha
A1 = np.diag(diag_prin, 0) + np.diag(diag_alphas, 1) + np.diag(diag_alphas, -1)

T = np.zeros((N,N))
T0_trans = np.transpose(T0)
T[:,0] = T0_trans

for i in range(N-1):
    T[:,i+1] = np.matmul(A1, T[:,i])


fig = plt.figure()
ax = fig.gca(projection='3d')
X, Time = np.meshgrid(x, t)

surf = ax.plot_surface(X, Time, T, rstride=1, cstride=1, antialiased=True)
print(alpha)
plt.show()
