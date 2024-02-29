import numpy as np
import random

#print((1 + np.exp(-1 * (-2))) ** - 1)


# for i in range(10):
#     print(random.choices(['brown', 'green'], weights=[9, 1])[0])

#print(12//2, 11/2)


cost_green_values = np.linspace(0.01, 0.5, num=10) 
for i in cost_green_values:
    print(i)