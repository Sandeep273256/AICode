import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score


auto = pd.read_csv("Auto.csv", na_values="?").dropna()
print(auto.head())

X = auto.drop(columns=["mpg", "name", "origin"])
y = auto["mpg"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

alphas = np.logspace(-3, 2, 50)
r2_ridge = []
r2_lasso = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    r2_ridge.append(ridge.score(X_test, y_test))

    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train, y_train)
    r2_lasso.append(lasso.score(X_test, y_test))

plt.plot(alphas, r2_ridge, label='Ridge')
plt.plot(alphas, r2_lasso, label='Lasso')
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('R2 Score')
plt.title('R2 Score vs Alpha')
plt.legend()
plt.show()

best_alpha_ridge = alphas[np.argmax(r2_ridge)]
best_alpha_lasso = alphas[np.argmax(r2_lasso)]  #

print("Best Ridge alpha:", best_alpha_ridge, "R2:", max(r2_ridge))
print("Best Lasso alpha:", best_alpha_lasso, "R2:", max(r2_lasso))

