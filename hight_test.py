import matplotlib.pyplot as plt
import numpy as np

x_values = np.linspace(0, 6000, 1000)

y_values = x_values*0.01 + 1
# plt.plot(x_values, y_values, "r")

y_values = np.log2(y_values) + 1
plt.plot(x_values, y_values, "g")

y_values = np.log10(y_values) * 30
plt.plot(x_values, y_values, "b")


# Отображаем сетку
plt.grid(True)

# Показываем график
plt.show()
