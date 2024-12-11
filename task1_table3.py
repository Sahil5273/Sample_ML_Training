import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Load the dataset
file_path = 'data.XLSX'
sheet_name = 'Table3_Alcohol data'
data = pd.read_excel(file_path, sheet_name=sheet_name)

# Remove unnamed columns
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

# Separate numeric and non-numeric columns
numeric_cols = data.select_dtypes(include=[np.number]).columns
non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns

# Handle missing values in numeric columns
imputer_num = SimpleImputer(strategy="mean")
data[numeric_cols] = imputer_num.fit_transform(data[numeric_cols])

# Convert all non-numeric columns to string to avoid mixed-type issues
for col in non_numeric_cols:
    data[col] = data[col].astype(str)

# Handle missing values in non-numeric columns
imputer_cat = SimpleImputer(strategy="most_frequent")
data[non_numeric_cols] = imputer_cat.fit_transform(data[non_numeric_cols])

# Encode categorical variables
label_encoders = {}
for col in non_numeric_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Separate features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Determine task type and initialize the model
if y.nunique() > 2:
    model = LinearRegression()
    task = "regression"
else:
    model = LogisticRegression(max_iter=1000)
    task = "classification"

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
if task == "regression":
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    metric = f"RMSE: {rmse:.4f}"
else:
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    f1 = f1_score(y_test, np.round(y_pred))
    metric = f"F1 Score: {f1:.4f}"

# Save visualizations to a PDF
with PdfPages('visualizations_table3.pdf') as pdf:
    # Descriptive statistics
    statistics = data.describe()
    plt.figure(figsize=(10, 6))
    plt.title("Descriptive Statistics")
    statistics.transpose().plot(kind='bar', figsize=(10, 6))
    plt.text(0.95, 0.95, metric, fontsize=12, transform=plt.gcf().transFigure, ha='right')
    pdf.savefig()
    plt.close()

    if task == "regression":
        # Residual plot
        residuals = y_test - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, residuals)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.title("Residual Plot")
        plt.text(0.95, 0.95, metric, fontsize=12, transform=plt.gcf().transFigure, ha='right')
        pdf.savefig()
        plt.close()
    else:
        # Confusion matrix
        cm = confusion_matrix(y_test, np.round(y_pred))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title("Confusion Matrix")
        plt.text(0.95, 0.95, metric, fontsize=12, transform=plt.gcf().transFigure, ha='right')
        pdf.savefig()
        plt.close()

print("Visualizations saved as 'visualizations_table3.pdf'.")
