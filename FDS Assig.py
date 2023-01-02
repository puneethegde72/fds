import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data=pd.read_csv("inputdata5.csv")
print(data)

x= data.iloc[:,0].values.reshape(-1, 1)
y= data.iloc[:,1].values.reshape(-1, 1)

model = LinearRegression()
model.fit(x,y)

y_pred=model.predict(x)

r=np.array(x.reshape(-1,1), dtype= 'float')

x_pred=260
y_pred1= model.predict([[x_pred]])
print(y_pred1)


plt.scatter(x, y)
plt.plot(x, y_pred, color='black')
plt.scatter(x_pred, y_pred1, color='Red')
plt.xlabel("Rainfall")
plt.ylabel("Productivity")
plt.show()

