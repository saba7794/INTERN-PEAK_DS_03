#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

# Load dataset from the uploaded file
data_path = 'path_to_boston_housing_data.csv'
data = pd.read_csv(data_path)

# Inspect the first few rows
data.head()


# In[4]:


import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Assume 'data' is your DataFrame
data = pd.DataFrame({
    'CRIM': [0.02731, 0.02729, None, 0.03237, 0.06905],
    'ZN': [18.0, 0.0, 0.0, None, 12.5],
    'INDUS': [2.31, 7.07, 7.07, 2.18, 2.18],
    'CHAS': ['0', '0', '0', '1', '0'],
    'RM': [6.575, 6.421, None, 7.147, 6.012],
    'AGE': [65.2, 78.9, 61.1, 45.8, 54.2],
    'MEDV': [24.0, 21.6, 34.7, 33.4, 36.2]
})

# Separate numeric and categorical data
numeric_features = data.select_dtypes(include=['number']).columns
categorical_features = data.select_dtypes(include=['object']).columns

# Handle missing values for numeric data
imputer = SimpleImputer(strategy='median')
data_numeric_imputed = pd.DataFrame(imputer.fit_transform(data[numeric_features]), columns=numeric_features)

# Handle missing values for categorical data (if any)
# For example, you can use the 'most_frequent' strategy
imputer_categorical = SimpleImputer(strategy='most_frequent')
data_categorical_imputed = pd.DataFrame(imputer_categorical.fit_transform(data[categorical_features]), columns=categorical_features)

# Combine the imputed numeric and categorical data
data_imputed = pd.concat([data_numeric_imputed, data_categorical_imputed], axis=1)

# Encode categorical variables (if any)
encoder = OneHotEncoder(drop='first')
data_encoded = pd.get_dummies(data_imputed, columns=categorical_features, drop_first=True)

# Feature scaling
scaler = StandardScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data_encoded), columns=data_encoded.columns)

# Now data_scaled is ready for further processing
print(data_scaled)


# In[5]:


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Check for missing values
missing_values = data.isnull().sum()

# Handle missing values
imputer = SimpleImputer(strategy='median')
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Encode categorical variables (if any)
# Assume 'CHAS' is a categorical feature in the Boston dataset
encoder = OneHotEncoder(drop='first')
categorical_features = ['CHAS']
data_encoded = pd.get_dummies(data_imputed, columns=categorical_features, drop_first=True)

# Feature scaling
scaler = StandardScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data_encoded), columns=data_encoded.columns)


# In[7]:


import seaborn as sns
import matplotlib.pyplot as plt

# Check the columns in data_scaled
print(data_scaled.columns)

# Assuming 'MEDV' is the target variable
target_variable = 'MEDV'

# Explore the distribution of the target variable
sns.histplot(data_scaled[target_variable], kde=True)
plt.title('Distribution of House Prices')
plt.xlabel('House Prices (in $1000s)')
plt.show()

# Correlation matrix
plt.figure(figsize=(12, 8))
correlation_matrix = data_scaled.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# In[8]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Split data
X = data_scaled.drop('MEDV', axis=1)
y = data_scaled['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)

# Random Forest
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Evaluation Metrics
def evaluate_model(predictions, true_values):
    rmse = mean_squared_error(true_values, predictions, squared=False)
    r2 = r2_score(true_values, predictions)
    return rmse, r2

# Evaluate Linear Regression
lr_rmse, lr_r2 = evaluate_model(lr_predictions, y_test)
print(f"Linear Regression RMSE: {lr_rmse}, R-squared: {lr_r2}")

# Evaluate Random Forest
rf_rmse, rf_r2 = evaluate_model(rf_predictions, y_test)
print(f"Random Forest RMSE: {rf_rmse}, R-squared: {rf_r2}")


# In[9]:


# Compare Models
model_comparisons = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest'],
    'RMSE': [lr_rmse, rf_rmse],
    'R-squared': [lr_r2, rf_r2]
})

print(model_comparisons)


