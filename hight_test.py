import matplotlib.pyplot as plt
import numpy as np

x_values = np.linspace(0, 6000, 1000)

# №1
h1 = np.log10(np.log2(x_values*0.01 + 1) + 1) * 30
plt.plot(x_values, h1, "r", alpha=0.5)

# №2
h2 = np.log2(x_values*0.003 + 1) * 7
plt.plot(x_values, h2, "g", alpha=0.5)

# №3 - последний используемый
h3 = np.log2(x_values*0.005 + 1)
h3 = h3 / h3[999] * 25
plt.plot(x_values, h3, "black")



x_ticks = np.arange(0, 6000, 200)  # Шаг 200
y_ticks = np.arange(0, 40, 5)

plt.xticks(x_ticks)
plt.yticks(y_ticks)
plt.grid(True)
plt.show()
