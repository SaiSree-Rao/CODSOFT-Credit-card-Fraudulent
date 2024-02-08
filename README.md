# CODSOFT-Credit-card-Fraudulent
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Loading the dataset
data = pd.read_csv("your_dataset.csv")

# Separating the features & target variable
X = data.drop(columns=['Label'])
y = data['Label']

# Splitting the dataset into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing: Standardizing the  features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handling class imbalance.
oversampler = RandomOverSampler(sampling_strategy='minority')
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train_scaled, y_train)

undersampler = RandomUnderSampler(sampling_strategy='majority')
X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train_resampled, y_train_resampled)

# Training a classification algorithm.
# starting with Logistic Regression.
model_lr = LogisticRegression()
model_lr.fit(X_train_resampled, y_train_resampled)

# Alternatively,we can try Random Forest classifier
# model_rf = RandomForestClassifier()
# model_rf.fit(X_train_resampled, y_train_resampled)

# Evaluate the model
y_pred = model_lr.predict(X_test_scaled)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
