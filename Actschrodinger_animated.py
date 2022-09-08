import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


N = 80                        # Número de puntos
L = 0.5                       # Longitud del espacio
tao = 0.001                   # Paso en tiempo
tf = (N-1)*tao                # Tiempo final
kappa = 5                # Coeficiente de difusión
t = np.linspace(0,tf,N)          # Vector de Tiempo
h = L/(N-1)                   # Paso en espacio
x = np.linspace(-L/2,L/2,N)      # Vector distancia
alpha = kappa*tao/(h**2)         # Parámetro de propagación
h_plank = 6.626 * 10**(-34)
mass = 1 * 10 **(-33)   #masa de la particula
ima = complex(0,1) #raiz de -1
IM = np.identity(N) #matriz identidad
k_0 = (2*math.pi)

# Gaussiana

sigma = 0.1
x0 = 0
#T0 = (1 / (sigma * math.sqrt(2*math.pi))) * np.exp(-1 * np.square(x - x0) / (2*(sigma**2)))
T0 = (1 / (sigma * math.sqrt(2*math.pi))) * np.exp(ima*k_0*x) * np.exp(-1 * np.square(x - x0) / (2*(sigma**2)))

## hamiltoniano

diag_prin = np.ones(N, dtype=complex) * (-2)
diag_alphas = np.ones(N-1, dtype=complex) * 1
ham = (-1)* (h_plank**2)/(2*mass) * ((np.diag(diag_prin, 0) + np.diag(diag_alphas, 1) + np.diag(diag_alphas, -1)) / h**2)

matrix = IM - ((ima*tao) / h_plank) * ham
matrix_inv = np.linalg.inv(IM + ((ima*tao) / h_plank) * ham) 

T = np.zeros((N,N), dtype=complex)
T0_trans = np.transpose(T0)
T[:,0] = T0_trans

for i in range(N-1):
    T[:,i+1] = np.matmul(np.matmul(matrix_inv, matrix),T[:,i])


def animation_handler(x, T):
    fig = plt.figure()
    ax = plt.axes(xlim=(-L/2, L/2), ylim=(0, 5))
    line, = ax.plot([], [], lw=3)

    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        y = T[:, i]
        line.set_data(x, y)
        return line,

    anim = FuncAnimation(fig, animate, init_func=init, frames=N, interval=200, blit=True)

    anim.save('test.gif', writer='imagemagick')

    plt.show()

def ThreeD_graph_handler(x, T):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Time = np.meshgrid(x, t)

    surf = ax.plot_surface(X, Time, np.abs(T), rstride=1, cstride=1, antialiased=True)

    plt.show()

animation_handler(x, T)
ThreeD_graph_handler(x, T)