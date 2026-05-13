import pandas as pd

data = pd.read_csv("train.csv")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

#print(data.keys())
print(data.info())
data = data[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'SalePrice']]
# Features (inputs)
X = data[['GrLivArea', 'BedroomAbvGr', 'FullBath']]

# Target (output)
y = data['SalePrice']

# Print features
print(X.head())

# Print target
print(y.head())

#print(data.isnull().sum())

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(X_train.shape)
print(y_train.shape)

#model formation
model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(predictions[:5])
print(y_test.head())
score = r2_score(y_test, predictions)
print("R2 scores :",score)

#visualization part
plt.scatter(y_test, predictions)

# Labels
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")

# Title
plt.title("Actual vs Predicted House Prices")

plt.show()