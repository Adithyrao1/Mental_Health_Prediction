import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load and preprocess the data
df = pd.read_csv('Deepression.csv')
df = df.drop(columns=['Number '], axis=1, errors='ignore')
df = df.dropna()

# Encode the target variable
label_encoder = LabelEncoder()
df['Depression State'] = label_encoder.fit_transform(df['Depression State'].values.ravel())

# Separate features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the parameter grid
param_grid = {
    'n_estimators': [100],
    'max_depth': [10, 11, 12, 13, 14, 15, 16, 30, 50, 90, 100],
    'criterion': ['squared_error', 'absolute_error', 'friedman_mse', 'poisson']
}

# Train the model with GridSearchCV
rf = RandomForestRegressor()
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)
best_rf = grid_search.best_estimator_

# Save the trained model and scaler
joblib.dump(best_rf, 'random_forest_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
