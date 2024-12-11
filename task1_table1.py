import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

file_path = 'data.XLSX'
sheet_name = 'Table1_Food data_NAm sources'
data = pd.read_excel(file_path, sheet_name=sheet_name)

data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

numeric_cols = data.select_dtypes(include=[np.number]).columns
non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns

imputer_num = SimpleImputer(strategy="mean")
data[numeric_cols] = imputer_num.fit_transform(data[numeric_cols])

imputer_cat = SimpleImputer(strategy="most_frequent")
data[non_numeric_cols] = imputer_cat.fit_transform(data[non_numeric_cols])

label_encoders = {}
for col in non_numeric_cols:
    data[col] = data[col].astype(str)
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if y.nunique() > 2:
    model = LinearRegression()
    task = "regression"
else:
    model = LogisticRegression(max_iter=1000)
    task = "classification"

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

if task == "regression":
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    metric = f"RMSE: {rmse:.4f}"
else:
    f1 = f1_score(y_test, np.round(y_pred))
    metric = f"F1 Score: {f1:.4f}"

with PdfPages('visualizations_table1.pdf') as pdf:
    statistics = data.describe()
    plt.figure(figsize=(10, 6))
    plt.title("Descriptive Statistics")
    statistics.transpose().plot(kind='bar', figsize=(10, 6))
    plt.text(0.95, 0.95, metric, fontsize=12, transform=plt.gcf().transFigure, ha='right')
    pdf.savefig()
    plt.close()

    if task == "regression":
        residuals = y_test - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, residuals)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.title("Residual Plot")
        plt.text(0.95, 0.95, metric, fontsize=12, transform=plt.gcf().transFigure, ha='right')
        pdf.savefig()
        plt.close()
    else:
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        cm = confusion_matrix(y_test, np.round(y_pred))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title("Confusion Matrix")
        plt.text(0.95, 0.95, metric, fontsize=12, transform=plt.gcf().transFigure, ha='right')
        pdf.savefig()
        plt.close()

print("Visualizations saved as 'visualizations1.pdf'.")
