import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score


from sklearn.datasets import load_diabetes

data = load_diabetes(as_frame=True)
df = data['frame']
print(data.DESCR)
print(df.head())

plt.hist(df["target"], 25)
plt.xlabel("target")
plt.title("Target Distribution")
plt.show()

sns.heatmap(df.corr().round(2), annot=True)
plt.title("Correlation Matrix")
plt.show()

plt.subplot(1, 2, 1)
plt.scatter(df['bmi'], df['target'])
plt.xlabel('bmi')
plt.ylabel('target')

plt.subplot(1, 2, 2)
plt.scatter(df['s5'], df['target'])
plt.xlabel('s5')
plt.ylabel('target')
plt.tight_layout()
plt.show()

X_base = df[['bmi', 's5']]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X_base, y, test_size=0.2, random_state=5)

model = LinearRegression()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

rmse_base = np.sqrt(mean_squared_error(y_train, y_train_pred))
r2_base = r2_score(y_train, y_train_pred)
rmse_base_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2_base_test = r2_score(y_test, y_test_pred)

print("Base RMSE Train:", rmse_base, "R2:", r2_base)
print("Base RMSE Test:", rmse_base_test, "R2:", r2_base_test)



X_bp = df[['bmi', 's5', 'bp']]

X_train, X_test, y_train, y_test = train_test_split(X_bp, y, test_size=0.2, random_state=5)

model.fit(X_train, y_train)
y_test_pred = model.predict(X_test)

rmse_bp = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2_bp = r2_score(y_test, y_test_pred)

print("With bp RMSE Test:", rmse_bp, "R2:", r2_bp)



X_all = df.drop(columns=["target"])
X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=0.2, random_state=5)

model.fit(X_train, y_train)
y_test_pred = model.predict(X_test)

rmse_all = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2_all = r2_score(y_test, y_test_pred)

print("All Features RMSE Test:", rmse_all, "R2:", r2_all)

