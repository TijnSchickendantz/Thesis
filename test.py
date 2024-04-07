import numpy as np
import random
import math
import matplotlib.pyplot as plt

#print((1 + np.exp(-1 * (-0.3))) ** - 1)


rat5C = dict()
numbs = [1,2,2,3]
for n,i in enumerate(numbs):
    if i not in rat5C:  # Make sure to add only 1 value per beta value and not overwrite
        rat5C[i] = n

print(rat5C)
#print((math.tanh(10/2*(-0.1))+1)/2)
#print(0/0.001)
# beta_vals = np.linspace(0,1,11)
# gamma_vals = np.linspace(0,1,11)
# adoption_J1P = np.zeros((len(gamma_vals), len(beta_vals)))
# for i, beta in enumerate(beta_vals):
#     for j, gamma in enumerate(gamma_vals):
#         adoption_J1P[j,i] = (beta+gamma)

# adoption_J1P = np.flipud(adoption_J1P)
# plt.imshow(adoption_J1P, cmap='gray_r', extent=[0, 1, 0, 1])
# plt.show()

# beta_vals, gamma_vals = np.meshgrid(np.linspace(0, 1, 11), np.linspace(0, 1, 11))

# # Calculate adoption_J1P directly using vectorized operations
# adoption_J1P = beta_vals + gamma_vals
# adoption_J1P = np.flipud(adoption_J1P)

# # Plotting
# plt.imshow(adoption_J1P, cmap='gray_r', extent=[0,1,0,1])
# plt.colorbar(label='Adoption')
# plt.xlabel('Beta')
# plt.ylabel('Gamma')
# plt.title('Adoption Plot')
# plt.show()
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