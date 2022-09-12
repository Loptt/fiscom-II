import random
import matplotlib.pyplot as plt

N = 3000
area_cuadrado = 4
radio_circulo = 1 
na = 0
xs = []
ys = []
cat = []

#generacion de numeros aleatorios de -1 a 1
for i in range(N):
    x = (random.random() * 2) - 1
    y = (random.random() * 2) - 1
    #aplicamos la condicion con los colores especificados
    if x**2 + y**2 < 1:
        na = na + 1
        cat.append('blue')
    else:
        cat.append('red')
    
    xs.append(x)
    ys.append(y)
#formula
A = na/N * area_cuadrado

print(A)

fig, ax = plt.subplots()

ax.scatter(xs, ys, c=cat)

plt.show()
