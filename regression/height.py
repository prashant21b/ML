import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , mean_absolute_error ,r2_score

df = pd.read_csv(r'regression\weight_height.csv')  


print(df)

plt.scatter(df['Height'],df['Weight'])

# plt.show()

x=df.corr()
print(x)
X=df[['Height']]
Y=df[['Weight']]
print(X)
print(Y)
scaler=StandardScaler()
X=scaler.fit_transform(X)
# Y=scaler.fit_transform(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


print(X_train)
print(Y_train)
print(X_test)
print(Y_test)

model=LinearRegression()
trained_model=model.fit(X_train,Y_train)
print(trained_model.coef_,model.intercept_)

predicted_values=trained_model.predict(X_test)

plt.plot(predicted_values,Y_test)
plt.show()
print("Predicted",predicted_values)

mse=mean_squared_error(Y_test,predicted_values)
print(mse)

mae=mean_absolute_error(Y_test,predicted_values)
print(mae)

r2=r2_score(Y_test,predicted_values)
print(r2)







