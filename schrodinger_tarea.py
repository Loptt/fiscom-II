import cmath as math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

N = 100                        # Número de puntos
M = 1600                       # Numero de timesteps
L = 1                       # Longitud del espacio
tao = 0.000005                   # Paso en tiempo
tf = (M-1)*tao                # Tiempo final
t = np.linspace(0,tf,M)          # Vector de Tiempo
h = L/(N-1)                   # Paso en espacio
x = np.linspace(0,L,N)      # Vector distancia
h_plank = 6.626 * 10**(-34)
mass = 1 * 10 **(-33)   #masa de la particula
ima = complex(0,1) #raiz de -1
IM = np.identity(N) #matriz identidad
k_0 = 250
sigma = math.sqrt(0.001)

c = (2/(math.pi*sigma**2))**(0.25)

E = ((h_plank * k_0) **2) / (2 * mass)

# Gaussiana

x0 = 0.4
#T0 = (1 / (sigma * math.sqrt(2*math.pi))) * np.exp(-1 * np.square(x - x0) / (2*(sigma**2)))
T0 = c * np.exp(-1 * np.square(x - x0) / (sigma**2)) * np.exp(ima*k_0*x)

# Potencial

# generate_potencial_barrera
# Genera un vector potencial donde el potencial es p entre x_1 y x_2 y 0 en otro caso
# p (float): valor del potencial
# x_1 (float): valor inicial donde hay potencial
# x_2 (float): valor final donde hay potencial
# return: un arreglo numpy 1D con los potenciales
def generate_potencial_barrera(p, x_1, x_2):
    v = np.zeros(shape=(N))
    booleans = np.array([a >= x_1 and a <= x_2 for a in x]) # Arreglo de booleanos donde b(i) es false si x < 0.6 si no es true.
    v = np.array([p if b else 0 for b in booleans]) # Usar el arreglo de booleanos donde v(i) es E cuando b(i) es true, si no es 0

    return v

def generate_potencial_oscilador(p, x, x0):
    v = np.array(12 * p * np.square(x - x0))
    return v

def generate_potencial_lineal(p, x):
    v = np.array(p * x)
    return v

# Como usar el potencial correcto para cada inciso:
#
# Para los incisos del 3 al 7 descomentar la primera linea e introducir los valores
# correspondientes a la llamada a la funcion para que se cree la barrera en el rango
# correcto
#
# Para el inciso 8 descomentar la segunda linea
#
# Para el inciso 9 descomentar la tercera linea

#v = generate_potencial_barrera((-1) *(10**6), x_1=0.6, x_2=1)
v = generate_potencial_oscilador(10, x, x0)
#v = generate_potencial_lineal(10, x)

print(v)

## hamiltoniano

diag_prin = np.ones(N, dtype=complex) * (-2) + v
diag_ones = np.ones(N-1, dtype=complex) * 1
ham = (-1)* (h_plank**2)/(2*mass) * ((np.diag(diag_prin, 0) + np.diag(diag_ones, 1) + np.diag(diag_ones, -1)) / h**2)

matrix = IM - ((ima*tao) / h_plank) * ham
matrix_inv = np.linalg.inv(IM + ((ima*tao) / h_plank) * ham) 

T = np.zeros((N,M), dtype=complex)
T0_trans = np.transpose(T0)
T[:,0] = T0_trans

for i in range(M-1):
    T[:,i+1] = np.matmul(np.matmul(matrix_inv, matrix),T[:,i])


def animation_handler(x, T):
    fig = plt.figure()
    ax = plt.axes(xlim=(0, L), ylim=(0, 6))
    line, = ax.plot([], [], lw=3)

    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        y = T[:, i]
        line.set_data(x, y)
        return line,

    anim = FuncAnimation(fig, animate, init_func=init, frames=M, interval=20, blit=True)

    anim.save('test.gif', writer='imagemagick')

    plt.show()

def ThreeD_graph_handler(x, T):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Time = np.meshgrid(x, t)

    surf = ax.plot_surface(X, Time, np.abs(T), rstride=1, cstride=1, antialiased=True)

    plt.show()

def Imagesc_graph_handler(x, T):
    fig = plt.figure()
    ax = fig.gca(xlim=(0,L))

    ax.imshow(np.rot90(T.astype(dtype=np.float32)), extent=[0,1,0,1])

    plt.show()



#animation_handler(x, T)
#ThreeD_graph_handler(x, T)
Imagesc_graph_handler(x, T)
