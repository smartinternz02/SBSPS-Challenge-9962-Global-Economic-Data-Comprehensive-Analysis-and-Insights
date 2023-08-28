import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
pc = pd.read_csv("Book1.csv")
data = {
    'Country': pc["country"],
    'Tourists': pc["tourists_in_millions"],
    'Percentage_of_gdp': pc["percentage of gdp"],
    'Annual_income': pc["annual income"],
}
df = pd.DataFrame(data)
X = df[['Tourists', 'Annual_income']]
y = df['Percentage_of_gdp']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
predictions = model.predict(X)
df['Predicted_growth'] = predictions
print(df[['Country', 'Predicted_growth']])
plt.barh(df["Country"],predictions)
plt.ylabel("Country")
plt.xlabel("Country Growth in %")
