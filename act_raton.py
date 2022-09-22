import numpy as np
import matplotlib.pyplot as plt

#se crea matriz de probabilidades 
prob_matrix = np.array([
  [1/2,1/2,0,0,0,0,0,0,0],
  [1/4,1/4,1/4,0,1/4,0,0,0,0],
  [0,1/3,1/3,0,0,1/3,0,0,0],
  [0,0,0,1/3,1/3,0,1/3,0,0],
  [0,1/3,0,1/3,1/3,0,0,0,0],
  [0,0,1/2,0,0,1/2,0,0,0],
  [0,0,0,1/3,0,0,1/3,1/3,0],
  [0,0,0,0,0,0,1/3,1/3,1/3],
  [0,0,0,0,0,0,0,1/2,1/2]
])

prob_density = np.array([1,0,0,0,0,0,0,0,0])

#variables
n_states = 9
n_ratones = 100
n_pasos = 100
final_state = 8

# se crean los ratones en el estado inicial
def setup_ratones():
  ratones = []
  for i in range(n_ratones):
    ratones.append(0)

  return ratones

#casilla a la que el raton se va a mover
def get_choice(raton):
  prob_array = np.array(prob_matrix[raton])
  choice = np.random.choice([x for x in range(n_states)], p=prob_array)
  return choice

#los pasos que da en raton en el trayecto
def run_n_steps(steps):
  global prob_density
  ratones = setup_ratones()

  for i in range(n_pasos):
    prob_density = np.matmul(prob_density, prob_matrix)
    for j in range(n_ratones):
      ratones[j] = get_choice(ratones[j]) 

  #print(ratones)
  #print(prob_density)
  #print("SUM: ", np.sum(prob_density))

  ##creacion de todos los histogramas
  
  plt.subplot(2,1,1)
  plt.title("Distribucion de ratones a 100 pasos")
  plt.hist(ratones, bins=9)

  plt.subplot(2,1,2)
  plt.title("PDF despues de 100 pasos")
  plt.bar(x=[i*0.8 for i in range(9)], height=prob_density, align='edge')

  plt.show()

#hace que todos los ratones lleguen al queso   
def run_indefinitely():
  ratones = setup_ratones()
  total_steps = [0 for x in range(n_ratones)]
  done = False

  while not done:
    done = True
    for i in range(n_ratones):
      if ratones[i] == final_state:
        continue
      
      ratones[i] = get_choice(ratones[i])
      total_steps[i] += 1

      if ratones[i] != final_state:
        done = False

  print(ratones)
  print(total_steps)

  plt.hist(total_steps, bins=10)
  plt.title("Histograma de pasos")
  plt.show()

run_n_steps(n_pasos)
#run_indefinitely()
