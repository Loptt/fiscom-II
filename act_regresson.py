import random
import math
import matplotlib.pyplot as plt
import numpy as np
import copy

# coordenadas iniciales
xs = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
ys = [1.49, 1.78, 1.95, 2.27, 2.49, 2.81, 2.89, 3.20, 3.45, 3.77, 3.97]
incertidumbre = 0.005
iterations = 2000

P = [0, 0]  # [A, B]


def M(P, x):
  return P[0] * x + P[1]

def calculate_x_2(xs, ys, incer, P):
  sum = 0
  for i in range(len(xs)):
    sum += ((ys[i] - M(P, xs[i])) / incer)**2
  return sum


def calculate_r(X_curr, X_cand):
  return (1 / (2 * math.pi)) * math.exp(-1 * 0.5 * (X_cand - X_curr))

def plot(xs, yx, P):
  plt.scatter(xs, ys)
  points = np.linspace(min(xs), max(xs), 100)
  lin_func = P[0] * points + P[1]

  plt.plot(points, lin_func)
  plt.show()


# evaluate_candidate_param evalua el un candidato nuevo de P dado por el
# parametro index
# e.g. en y = Ax + B
# A = 0 evalua el parametro A
# B = 1 evalua el parametro B
def evaluate_candidate_param(xs, ys, P, index, X_curr):
  # Generar un nuevo valor candidato
  cand = P[index] + random.gauss(mu=0, sigma=1)

  # Generar nuevo P con nuevo valor candidato
  P_cand = copy.deepcopy(P)
  P_cand[index] = cand

  # Calcula X candidato con nuevo P
  X_cand = calculate_x_2(xs, ys, incertidumbre, P_cand)

  if X_cand < X_curr:  # Reemplazar candidato immediatamente
    P[index] = P_cand[index]
    return X_cand
  else:  # Reemplazar candidato a la suerte
    R = calculate_r(X_curr, X_cand)
    r = random.random()
    if r <= R:
      P[index] = P_cand[index]
      return X_cand

  # Si nunca se cambio el candidato, regresar el valor de X_curr
  return X_curr
  

X_curr = calculate_x_2(xs, ys, incertidumbre, P)

for i in range(iterations):
  # Generar una probabilidad de 0.5 de escoger A o B
  if random.random() > 0.5:  # Escoger A
    X_curr = evaluate_candidate_param(xs, ys, P, 0, X_curr)
  else:  # Escoger B
    X_curr = evaluate_candidate_param(xs, ys, P, 1, X_curr)

plot(xs, ys, P)
