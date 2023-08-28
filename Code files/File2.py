import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
pc = pd.read_csv("Book1.csv")
pc.head()
data = {
    'Country': pc["country"],
    'Percentage_of_gdp': pc["percentage of gdp"],
    'Cost_index': pc["cost_index"],
    'Purchasing_power_index': pc["purchasing power index"],
}
df = pd.DataFrame(data)
X = df[['Cost_index', 'Purchasing_power_index']]
y = df['Percentage_of_gdp']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
predictions = model.predict(X*10)
df['Predicted_growth'] = predictions
print(df[['Country', 'Predicted_growth']])
plt.barh(df["Country"],predictions)
plt.xlabel("Country")
plt.ylabel("Country's growth in %")