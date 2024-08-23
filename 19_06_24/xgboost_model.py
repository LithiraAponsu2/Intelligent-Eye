import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np  # Import NumPy for ndarray handling
import joblib  # Import joblib for model saving

# Load the CSV file into a DataFrame
csv_file_path = "combined_csv.csv"  # Replace with the path to your CSV file
df = pd.read_csv(csv_file_path)

# Assuming the last column is the target variable, and the rest are features
X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]

# Print number of input features
num_features = X.shape[1]
print("Number of input features:", num_features)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert pandas DataFrames/Series to numpy arrays
X_train_np = X_train.values
y_train_np = y_train.values

# Create and train the XGBoost model
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train_np, y_train_np)

# Save the trained model
model_file = "xgb_model.pkl"  # Name your model file
joblib.dump(xgb_model, model_file)

print(f"Saved model to {model_file}")

# Make predictions on the test set
y_pred = xgb_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, xgb_model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(classification_rep)

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
