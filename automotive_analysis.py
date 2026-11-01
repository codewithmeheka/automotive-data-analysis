```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Load the automotive dataset
url = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "auto-mpg/auto-mpg.data"
)

columns = [
    "mpg", "cylinders", "displacement", "horsepower",
    "weight", "acceleration", "model_year", "origin", "car_name"
]

df = pd.read_csv(
    url,
    sep=r"\s+",
    names=columns,
    na_values="?",
    comment="\t",
    quotechar='"',
    encoding="latin-1"
)

# Step 2: Clean the data
df = df.dropna()
df = df.drop_duplicates()

print("First five records:")
print(df.head())

print("\nDataset information:")
print(df.describe())

# Step 3: Visualize fuel efficiency
plt.figure(figsize=(8, 5))
sns.histplot(df["mpg"], kde=True)
plt.title("Distribution of Vehicle Fuel Efficiency")
plt.xlabel("Miles Per Gallon (MPG)")
plt.ylabel("Number of Vehicles")
plt.tight_layout()
plt.savefig("fuel_efficiency_distribution.png")
plt.show()

# Step 4: Explore vehicle weight and fuel efficiency
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="weight", y="mpg")
plt.title("Vehicle Weight vs Fuel Efficiency")
plt.xlabel("Vehicle Weight")
plt.ylabel("Miles Per Gallon (MPG)")
plt.tight_layout()
plt.savefig("weight_vs_mpg.png")
plt.show()

# Step 5: Prepare data for machine learning
X = df[
    [
        "cylinders", "displacement", "horsepower",
        "weight", "acceleration", "model_year", "origin"
    ]
]
y = df["mpg"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 6: Train the prediction model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Make predictions
predictions = model.predict(X_test)

# Step 8: Evaluate the model
mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("\nModel Evaluation Results")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R-squared Score: {r2:.2f}")

# Step 9: Display sample predictions
results = pd.DataFrame({
    "Actual MPG": y_test.values,
    "Predicted MPG": predictions.round(2)
})

print("\nSample Predictions:")
print(results.head(10))

# Step 10: Save predictions
results.to_csv("automotive_predictions.csv", index=False)

print("\nAnalysis completed successfully!")
print("Graphs and predictions have been saved.")
```
