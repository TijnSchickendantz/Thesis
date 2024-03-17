import numpy as np
import random
import math

print((1 + np.exp(-5 * (-0.2))) ** - 1)
#print((math.tanh(10/2*(-0.1))+1)/2)
#print(0/0.001)

# for i in range(10):
#     print(random.choices(['brown', 'green'], weights=[9, 1])[0])

#print(12//2, 11/2)


# cost_green_values = np.linspace(0, 1, 11) 
# for i,g in enumerate(cost_green_values):
#     print(g)

# beta_vals = np.linspace(0,1,11)
# gamma_vals = np.linspace(0,1,11)
# adoption_J1P = np.zeros((len(gamma_vals), len(beta_vals)))

# for i, beta in enumerate(beta_vals):
#         for j, gamma in enumerate(gamma_vals):
#                 adoption_J1P[j,i] = gamma

# #flipped_array1 = np.flip(adoption_J1P, axis=1)
# adoption_J1P = np.flipud(adoption_J1P)
# print(adoption_J1P)
# #print(flipped_array1)