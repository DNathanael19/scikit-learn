import numpy as np 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
x = np.array([[0], [5], [2], [15], [20]])
y = np.array([0, 10, 20, 30, 40])
reg = LinearRegression().fit(x,y)
#reg.score(x,y)
print(reg.coef_)
print(reg.intercept_)
#reg.predict(np.array([[3,5]]))

plt.scatter(x,y)

a = np.linspace(0, 20, 100)
b = reg.coef_* a + reg.intercept_
plt.plot(a, b, 'r-')
plt.show()