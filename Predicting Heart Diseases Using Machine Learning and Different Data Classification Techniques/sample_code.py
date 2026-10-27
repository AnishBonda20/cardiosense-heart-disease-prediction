# Importing necessary libraries for data manipulation, analysis, and visualization: pandas, numpy, matplotlib, and seaborn.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, QuantileTransformer, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')
# print heading - for display purposes only
def print_heading(heading):
    print('-' * 50)
    print(heading)
    print('-' * 50)
# read data
df = pd.read_csv('/kaggle/input/heart-disease-data/heart_disease_uci.csv')
df.head()
# read data
df = pd.read_csv("/kaggle/input/heart-disease-data/heart_disease_uci.csv")
df.head()
# Check the shape of the data
df.shape
print_heading('Shape of the data')
print(f"There are {df.shape[0]} patients and {df.shape[1]} features in the dataset.")
df.info()
print_heading("DATASET INFORMATION SUMMARY")

print("1. The dataset contains 920 entries across 16 features pertinent to heart disease studies.")
print("2. Complete data is available for 'id', 'age', 'sex', 'dataset', 'cp', and the outcome variable 'num'.")
print("3. Significant missing data is found in 'ca' with 66.41% missing.")
print("4. Significant missing data is found in 'thal' with 52.83% missing.")
print("5. 'Slope' also has a considerable amount of missing data, with only 611 non-null entries.")
print("6. Variables 'trestbps', 'chol', 'fbs', 'thalch', 'exang', and 'oldpeak' have some missing values but to a lesser extent.")
df.describe()
print_heading('Observations from summary statistics'.upper())

print(f"1. There are no heart diseases found in the 25% of the patients with an average of age {df.describe().loc['25%', 'age']}.")
print(f"2. There is mild presence of heart diseases found in the 50% of the patients with an average of age {df.describe().loc['50%', 'age']}.")
print(f"3. There is moderate presence of heart diseases found in the 75% of the patients with an average of age {df.describe().loc['75%', 'age']} or more.")
# Plot the heatmap on dataset to check the missing values 
plt.figure(figsize=(16, 6))
sns.heatmap(df.isnull(), cbar=False)
plt.title('Missing value in the dataset')
plt.show()
round(((df.isnull().sum() / len(df)) * 100).sort_values(ascending=False),2)
print_heading('Observations from missing values'.upper())

print("1. The dataset shows significant gaps in key features, with 'ca' and 'thal' being the most affected, missing 66.41% and 52.83% of their data, respectively.\n"
      "2. 'Slope' also has a substantial 33.59% of values missing.\n"
      "3. These high levels of incompleteness are likely to pose challenges for any predictive analysis.\n"
      "4. Advanced imputation techniques may be necessary to address the deficiencies before any reliable model training can be conducted.")
# Automatically find numerical and categorical columns with missing values less than 10%
missing_threshold = 10  # 10% threshold for missing values
total_entries = len(df)
target_variable = 'num'

# list of numerical columns that have less than a certain threshold of missing values and are not the target variable.
numerical_cols = [
    col 
    for col in df.columns 
    if (
        (df[col].dtype in ['int64', 'float64']) and 
        (df[col].isnull().sum() / total_entries * 100) < missing_threshold and 
        col != target_variable
    )
]

# list of numerical columns that have less than a certain threshold of missing values
categorical_cols = [col
                   for col in df.columns 
                   if (
                        df[col].dtype == 'object') and 
                        (df[col].isnull().sum() / total_entries * 100) < missing_threshold]

# Apply simple imputation to numerical and categorical columns
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

# Apply imputation to numerical and categorical columns
for col in numerical_cols:
    df.loc[:,col] = num_imputer.fit_transform(df[[col]])

for col in categorical_cols:
    df.loc[:, col] = cat_imputer.fit_transform(df[[col]])

print_heading('Observations from missing values after imputation'.upper())

# Confirm there are no missing values left

missing_values_sum = df[df.columns.difference(['thal', 'ca', 'slope'])].isnull().sum()

print(f"The sum of all missing values, excluding features ['thal', 'ca', 'slope'], is:\n{missing_values_sum}"  + "\n")
print_heading('High missing values features'.upper())

# Display the features with missing values 
print(f"The sum of all high missing values where simple imputation wasn't used, is:\n{df[['thal', 'ca', 'slope']].isnull().sum()}")
# create heatmap to check missing values after imputation
plt.figure(figsize=(16, 6))
sns.heatmap(df.isnull(), cbar=False)
plt.title('Missing values after imputation')
plt.show()
# split the dataset into two subset with one contaning complete data and the other containing missing data for thal, ca, and slope. these dataset should still have all the columns in the result.
# create a new dataframe which removes all the rows that nan values
df_clean = df.dropna()

# create a new dataframe which contains all the rows that have nan values
df_missing = df[df.isna().any(axis=1)]

print_heading('Complete data with no missing values'.upper())
print(f"The shape of the complete dataframe is: {df_clean.shape}")
print("\n")
print_heading('Missing data with high missing values'.upper())
print(f"The shape of the dataframe with missing values is: {df_missing.shape}")
# Training and evaluating the model for the target variables 'slope', 'thanl', and 'ca'.
def predict_and_evaluate(target_variable, drop_columns):
    # Drop specified columns from the DataFrame
    X = df_clean.drop(drop_columns, axis=1)
    
    # Select target variable(s)
    y = df_clean[target_variable]

    # Initialize LabelEncoder
    le = LabelEncoder()
    
    #le = OneHotEncoder()
    for col in X.columns:
        # If column is of object type
        if X[col].dtype == 'object':
            # Encode the column
            X[col] = le.fit_transform(X[col])
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize RandomForestClassifier
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Print heading
    print_heading(f'Model performance for {target_variable}'.upper())
    
    # Print accuracy score
    print(f"The accuracy of the model is: {round(accuracy_score(y_test, y_pred),2)}")

    # Print classification report
    print(f"Classification report:\n {classification_report(y_test, y_pred)}")

    # Return the trained model
    return model
# Train and evaluate a model for predicting the 'slope' feature, excluding 'slope', 'ca', and 'thal' from the input features.
rf_model_slope  = predict_and_evaluate('slope', ['slope', 'ca', 'thal'])
# Train and evaluate a model for predicting the 'ca' feature, excluding 'ca', and 'thal' from the input features.
rf_model_ca = predict_and_evaluate('ca', [ 'ca', 'thal'])
# # Train and evaluate a model for predicting the 'thal' feature.
rf_model_thal = predict_and_evaluate('thal', ['thal'])
# create a new dataframe and drop the target variables
df_encoded_pred = df_missing.drop(['slope', 'ca', 'thal'], axis=1)

# Encode categorical features and predict the missing values
def encode_and_predict(df, target_variable, model):

    # Initialize label encoder
    le = LabelEncoder()

    # Loop through each column in the DataFrame
    for col in df.columns:
        # Check if the column is of type 'object' (categorical)
        if df[col].dtype == 'object':
            # encode the categorical data to integers
            df[col] = le.fit_transform(df[col])
    # Use the model to apply predictions and assign the predicted values to the target variable
    df[target_variable] = model.predict(df)

# Encode categorical features and predict the 'slope' column in df_encoded_pred
encode_and_predict(df_encoded_pred, 'slope', rf_model_slope)
# Update the 'slope' column in df_missing with the predicted values
df_missing.loc[:,'slope'] = df_encoded_pred['slope']
Predicting missing values for ca - Number of major vessels (0-3) colored by fluoroscopy.
# Create a new dataframe and drop the target variables
df_encoded_pred = df_missing.drop(['ca', 'thal'], axis=1)
# Encode categorical features and predict the 'ca' column in df_encoded_pred
encode_and_predict(df_encoded_pred, 'ca', rf_model_ca)
# Update the 'ca' column in df_missing with the predicted values
df_missing.loc[:, 'ca'] = df_encoded_pred['ca']
Predicting missing values for thal - Thalassemia category.
# Create a new dataframe and drop the target variables
df_encoded_pred = df_missing.drop(['thal'], axis=1)
# Encode categorical features and predict the 'thal' column in df_encoded_pred
encode_and_predict(df_encoded_pred, 'thal', rf_model_thal)
# Update the 'thal' column in df_missing with the predicted values
df_missing.loc[:, 'thal'] = df_encoded_pred['thal']
Next step - Merging the datasets
# merge df_clean and df_missing 
df = pd.concat([df_clean, df_missing])

print_heading('Dataset shape inspect'.upper())
print(f"The shape of the dataset is: {df.shape[0]} rows and {df.shape[1]} columns.") # Display the missing values if any in the dataset for verification
print_heading('Missing values inspect'.upper())
if df.isnull().sum().sum() == 0:
    print("There are no missing values in the dataset.")
else:
    print("There are missing values in the dataset.")
# create a heatmap for df to check missing values
plt.figure(figsize=(16, 6))
sns.heatmap(df.isnull(), cbar=False)
plt.title('Missing values after imputation')
plt.show()
# Plotting the histogram and boxplot to check the data distribution for the numerical columns 
def plot_histogram_and_boxplot(df, numerical_cols, cyan_palette=["#00FFFF"]):
    # Set to "dark" for a black background
    sns.set(style="dark", palette="deep")

    # Set the figure size to accommodate the subplots
    plt.figure(figsize=(15, len(numerical_cols) * 5))

    # Loop through each numerical column and create a row with a histogram and a boxplot
    for i, col in enumerate(numerical_cols):
        # Histogram (left)
        ax1 = plt.subplot(len(numerical_cols), 2, 2*i + 1)
        sns.histplot(df[col], kde=True, bins=30, color=cyan_palette[0])
        plt.title(f'Histogram of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        ax1.set_facecolor('#000133')

        # Boxplot (right)
        ax2 = plt.subplot(len(numerical_cols), 2, 2*i + 2)
        sns.boxplot(x=df[col], color=cyan_palette[0], linewidth=2.5,
                    flierprops=dict(markerfacecolor='w', marker='o', markeredgecolor='w', markersize=5),
                    whiskerprops=dict(color='white', linewidth=2.5),
                    capprops=dict(color='white', linewidth=2.5),
                    boxprops=dict(edgecolor='white', linewidth=2.5))
        plt.title(f'Boxplot of {col}')
        ax2.set_facecolor('#000133')

    # Adjust layout for better readability and display the plot
    plt.tight_layout()
    plt.show()
# Remove id column from numerical_cols
numerical_cols.remove('id')
# Plot histogram and boxplot for numerical columns before scaling
plot_histogram_and_boxplot(df, numerical_cols)
# Instantiate the QuantileTransformer with a normal output distribution and a fixed random state for reproducibility
quantile_transformer = QuantileTransformer(random_state=42, output_distribution='normal')

# Apply the transformer to the numerical columns of the DataFrame
df[numerical_cols] = quantile_transformer.fit_transform(df[numerical_cols])
# Calculate quantile ranges for each numerical column
def calc_lower_upper_bounds(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    # Calculate the lower and upper bound for outliers which is 1.5 times the IQR less than the first and third quartile
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return lower_bound, upper_bound

# Detect outliers based on the IQR
def detect_outliers(df, column):
    
    lower_bound, upper_bound = calc_lower_upper_bounds(df, column)
    # Detect outliers based on the lower and upper bounds
    outliers_detected = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    #calculate the percentage of outliers in the dataset
    outliers_percentage = round(len(outliers_detected) / df[column].shape[0] * 100, 2)
    print(f"Number of outliers detected in {column}: {outliers_percentage}%")
    
# Remove outliers based on the IQR
def remove_outliers(df, column):
    
    lower_bound, upper_bound = calc_lower_upper_bounds(df, column)
    # Remove outliers from the dataframe
    df_no_outliers = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df_no_outliers

# Disaplay heading
print_heading(f"percentages of outliers detected".upper())

# Detect and remove outliers for the numerical column
for col in numerical_cols:
    detect_outliers(df, col)
    # Only remove outliers for 'chol'
    if col in ['chol']:
        df = remove_outliers(df, col)
# Plot histogram and boxplot for numerical columns after removing outliers
plot_histogram_and_boxplot(df, ['chol'])
# Drop the target column 'num' from the DataFrame to create the features dataset
X = df.drop('num', axis=1)

# Create the target dataset
y = df['num']

# Initialize LabelEncoder
le = LabelEncoder()
for col in X.columns:
    # If column is of object type
    if X[col].dtype == 'object':
        # Encode the column to convert categorical variables into numerical
        X[col] = le.fit_transform(X[col])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestClassifier model
rf_model = RandomForestClassifier(n_estimators=10, max_depth=10, random_state=42)

# Train the model on the training data
rf_model.fit(X_train, y_train)

# Use the trained model to predict the target variable for the testing data
y_pred = rf_model.predict(X_test)

# Compute the confusion matrix to evaluate the accuracy of a classification
cm = confusion_matrix(y_test, y_pred)

# Print the confusion matrix, accuracy of the model and the classification report
print_heading('Model performance'.upper())
print(f"The confusion matrix is:\n {confusion_matrix(y_test, y_pred)}")
print(f"\nThe accuracy of the model is: {round(accuracy_score(y_test, y_pred),2)}")
print(f"Classification report:\n {classification_report(y_test, y_pred)}")
# Create a heatmap from the confusion matrix
plt.figure(figsize=(15, 8))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')

# Adding labels to the heatmap
plt.title('Confusion Matrix')
plt.xlabel('Predicted labels')
plt.ylabel('True labels') 
plt.show()
