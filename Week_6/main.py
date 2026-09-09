import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


hours = np.array([1,2,3,4,5,6,7,8]).reshape(-1,1)
marks = np.array([25,35,45,55,65,75,85,95])

model = LinearRegression()
model.fit(hours, marks)


h = float(input("Enter study hours: "))
prediction = model.predict([[h]])

print("Predicted Marks =", round(prediction[0],2))


plt.scatter(hours, marks, label="Actual Data")
plt.plot(hours, model.predict(hours), label="Prediction Line")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.legend()
plt.show()