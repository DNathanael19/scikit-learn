import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
x = np.array([[0], [5], [2], [15], [20]])
y = np.array([0, 10, 20, 30, 40])
reg = LinearRegression().fit(x,y)
#reg.score(x,y)
print(reg.coef_)
print(reg.intercept_)
#reg.predict(np.array([[3,5]]))

meu_y = reg.coef_ * x + reg.intercept_
print('Meu erro:')
print(mean_squared_error(y, meu_y))
y_raul = 2 * x
print('Erro do Raul:')
print(mean_squared_error(y_raul, meu_y))

plt.scatter(x,y)

a = np.linspace(0, 20, 100)
b = reg.coef_* a + reg.intercept_
plt.plot(a, b, 'r-')
plt.show()