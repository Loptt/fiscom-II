import copy
import random
import matplotlib.pyplot as plt

x_max = 5
y_max = 5
x_min = -5
y_min = -5

step = 0.01
N = 10000
n_particulas = 1000
particulas = []

particulas_hist = {}

for i in range(n_particulas):
  particulas.append([0, 0])
  particulas_hist[i] = []

def mutate_position(num):
  if random.random() < 0.5:
    num += step
    if num > x_max:
      num = x_max
  else:
    num -= step
    if num < x_min:
      num = x_min

  return num


for i in range(N):
  for i, coord in enumerate(particulas):
    particulas_hist[i].append(copy.deepcopy(coord))
    coord[0] = mutate_position(coord[0])
    coord[1] = mutate_position(coord[1])

def show_history(history, n):
  if n > 3:
    print("Cannot show history for more than 3 particles")
    return

  colors = ["blue", "red", "green"]
  xs = []
  ys = []
  colors_scatter = []

  for i in range(n):
    xs = xs + [coord[0] for coord in history[i]]
    ys = ys + [coord[1] for coord in history[i]]
    colors_scatter = colors_scatter + [colors[i]] * len(history[i])

  fig = plt.figure(figsize=(4,3))
  ax = plt.axes()

  ax.scatter(xs, ys, c=colors_scatter)
  ax.yaxis.grid()
  ax.xaxis.grid()

  plt.show()


def show_histogram(particulas):
  xs = [coord[0] for coord in particulas]
  ys = [coord[1] for coord in particulas]

  plt.subplot(1,2,1)
  plt.hist(xs, bins=50)
  plt.title("X coordinate")

  plt.subplot(1,2,2)
  plt.hist(ys, bins=50)
  plt.title("Y coordinate")

  plt.show()

show_history(particulas_hist, 3)
show_histogram(particulas)
