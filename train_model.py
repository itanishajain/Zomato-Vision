import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('Zomato data.csv')  # Replace 'your_dataset.csv' with your actual file path

# Display the columns and a preview of the dataset
print(f"Columns in the dataset: {df.columns}")
print(df.head())

# Preprocess the dataset: Encode categorical features
label_encoder = LabelEncoder()

# Encode categorical columns if they exist
categorical_columns = ['online_order', 'book_table', 'listed_in(type)']
for col in categorical_columns:
    if col in df.columns:
        df[col] = label_encoder.fit_transform(df[col])

# Define features and target variable
# Assuming 'rate' is the target, replace with the actual target column if needed
X = df.drop(['name', 'rate'], axis=1)  # Drop non-numeric and target columns
y = df['rate']  # Replace with your target column

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestRegressor model
rf_regressor = RandomForestRegressor(random_state=42)

# Train the model
rf_regressor.fit(X_train, y_train)

# Make predictions
y_pred = rf_regressor.predict(X_test)

# Evaluate the model: Mean Squared Error (MSE) and R2 score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf_regressor, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Best parameters and score from GridSearchCV
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# Refit the model with the best parameters from GridSearchCV
best_rf_regressor = grid_search.best_estimator_

# Evaluate the tuned model
y_pred_tuned = best_rf_regressor.predict(X_test)
mse_tuned = mean_squared_error(y_test, y_pred_tuned)
r2_tuned = r2_score(y_test, y_pred_tuned)

print(f"Tuned Mean Squared Error: {mse_tuned}")
print(f"Tuned R2 Score: {r2_tuned}")