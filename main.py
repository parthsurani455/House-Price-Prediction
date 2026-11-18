import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Sample dataset
data = {
    'Area': [800, 1000, 1200, 1500, 1800, 2000,
             900, 1100, 1300, 1600, 1900, 2200],

    'Bedrooms': [2, 2, 3, 3, 3, 4,
                 2, 2, 3, 3, 4, 4],

    'Bathrooms': [1, 2, 2, 2, 3, 3,
                  1, 2, 2, 2, 3, 3],

    'Age': [10, 8, 5, 6, 4, 3,
            12, 9, 7, 5, 3, 2],

    'Price': [25, 32, 40, 48, 55, 65,
              28, 35, 43, 50, 60, 70]
}

df = pd.DataFrame(data)

# Input and output
X = df[['Area', 'Bedrooms', 'Bathrooms', 'Age']]
y = df['Price']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()

model.fit(X_train, y_train)

# Check model score
accuracy = model.score(X_test, y_test)

print("Model Training Completed!")
print("R2 Score:", accuracy)

# Save trained model
with open("house_price_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model Saved Successfully!")