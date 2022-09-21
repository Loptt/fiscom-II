import numpy as np

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

n_states = 9
n_ratones = 100
n_pasos = 100
final_state = 8

def setup_ratones():
  ratones = []
  for i in range(n_ratones):
    ratones.append(0)

  return ratones

def get_choice(raton):
  prob_array = np.array(prob_matrix[raton])
  choice = np.random.choice([x for x in range(n_states)], p=prob_array)
  return choice

def run_n_steps(steps):
  ratones = setup_ratones()

  for i in range(n_pasos):
    for j in range(n_ratones):
      if ratones[j] == final_state:
        continue
      ratones[j] = get_choice(ratones[j]) 

  print(ratones)

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

#run_n_steps(n_pasos)
run_indefinitely()
