
# Insurance Charges Regression Analysis (Python Script Version for Pydroid)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

data = pd.read_csv('insurance.csv')
print("Data Loaded Successfully!\n")
print(data.head())

print("Data Info:\n")
print(data.info())
print("\nData Description:\n")
print(data.describe())

print("\nMissing Values:\n")
print(data.isnull().sum())

sns.pairplot(data, hue='smoker')
plt.savefig('pairplot.png')
plt.close()

plt.figure(figsize=(8,6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.close()

plt.figure(figsize=(10,6))
sns.boxplot(data=data, x='smoker', y='charges')
plt.title('Charges vs Smoker Status')
plt.savefig('boxplot_smoker_charges.png')
plt.close()

print("Exploratory Data Analysis Completed. Plots saved as images.\n")

data_encoded = pd.get_dummies(data, drop_first=True)
X = data_encoded.drop('charges', axis=1)
y = data_encoded['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split into training and test sets.\n")

model = LinearRegression()
model.fit(X_train, y_train)
print("Model Training Completed.\n")

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

print(f'R^2 Score: {r2}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'MAE: {mae}\n')

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual Charges')
plt.ylabel('Predicted Charges')
plt.title('Actual vs Predicted Charges')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.savefig('actual_vs_predicted.png')
plt.close()

print("Model evaluation completed. Prediction plot saved as 'actual_vs_predicted.png'.\n")

joblib.dump(model, 'insurance_charges_model.pkl')
print("Model saved as 'insurance_charges_model.pkl'.\n")

print("\nAll tasks completed successfully!")
