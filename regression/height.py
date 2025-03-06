import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

df=pd.read_csv('weight_height.csv')

print(df)

plt.scatter(df['Height'],df['Weight'])

plt.show()

x=df.corr()
print(x)
X=df[['Height']]
Y=df[['Weight']]
print(X)
print(Y)

X_train, Y_trian, X_test,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

print(X_train)
print(Y_trian)
print(X_test)
print(Y_test)


