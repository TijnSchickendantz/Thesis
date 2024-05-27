import numpy as np
import random
import math
import matplotlib.pyplot as plt
import seaborn as sns

# vals = [0.43,0.5,0.45,0.39,0.45]
# print(np.mean(vals))
# print(np.std(vals))

#print(np.linspace(0,0.5,11))

# Generate example data (replace this with your simulation results)
# initial_conditions_j1 = np.linspace(0, 1, 10)
# initial_conditions_j2 = np.linspace(0, 1, 10)
# #average_adoption_rates = np.random.rand(10, 10)  # Example random data

# X, Y = np.meshgrid(initial_conditions_j1, initial_conditions_j2)

# Calculate the outcome for each initial condition
# outcomes = np.where(X < Y, 0, 1)  # Values to the left of the diagonal go to 0, and to the right go to 1

# plt.figure(figsize=(5, 5))

# # Plot initial condition values
# plt.scatter(X, Y, color='blue', label='Initial Conditions')

# # Draw arrows towards the outcome
# for i in range(len(initial_conditions_j1)):
#     for j in range(len(initial_conditions_j2)):
#         if outcomes[j, i] == 0:
#             plt.arrow(X[j, i], Y[j, i], 0.1, 0.1, color='red', head_width=0.03, head_length=0.05)
#         else:
#             plt.arrow(X[j, i], Y[j, i], -0.1, -0.1, color='red', head_width=0.03, head_length=0.05)

# plt.title('Phase Diagram: Initial Conditions and Outcome Arrows')
# plt.xlabel('Initial Condition for Jurisdiction 1')
# plt.ylabel('Initial Condition for Jurisdiction 2')
# plt.xlim(0, 1)
# plt.ylim(0, 1)
# plt.legend()
# plt.grid(True)
# plt.show()

# payoff = 0.3
# av=0.2
# rat=1

#print(1 - (1 + np.exp(-100 * (0.1))) ** - 1)

# def sigmoid_function(av_payoff, rat):
#     return (1 + np.exp(-rat * av_payoff)) ** -1

# # Generate x values (av_payoff) from -1 to 1
# x_values = np.linspace(-1, 1, 100)

# # Values of rat
# rat_values = [1, 10, 100]

# # Create the plot
# plt.figure(figsize=(6, 4))

# # Plot the function for each value of rat
# for rat in rat_values:
#     y_values = sigmoid_function(x_values, rat)
#     plt.plot(x_values, y_values, label=f'rat = {rat}')

# # Add labels and legend
# #plt.title('Sigmoid Function for Different Values of rat')
# plt.xlabel('payoff difference')
# plt.ylabel('Probability of switching')
# plt.legend()
# #plt.grid(True)
# plt.show()


# rat5C = dict()
# numbs = [1,2,2,3]
# for n,i in enumerate(numbs):
#     if i not in rat5C:  # Make sure to add only 1 value per beta value and not overwrite
#         rat5C[i] = n

# print(rat5C)


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