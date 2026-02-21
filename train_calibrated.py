# train_calibrated.py
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

print("Loading data and scaler...")
scaler = joblib.load("scaler.pkl")
df = pd.read_csv("vgsales.csv").dropna(subset=[
    "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"
])

df["Sales_Class"] = pd.qcut(df["Global_Sales"], q=3, labels=[0, 1, 2]).astype(int)

X_cal = df[["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]]
y_cal = df["Sales_Class"]

# Use a larger sample for better calibration since it's offline
X_sample = X_cal.sample(1500, random_state=42) 
y_sample = y_cal.loc[X_sample.index]
X_scaled = scaler.transform(X_sample)

print("Training Calibrated Model...")
svm_clone = SVC(probability=True, kernel="rbf", random_state=42)
calibrated_model = CalibratedClassifierCV(svm_clone, method="sigmoid", cv=3)
calibrated_model.fit(X_scaled, y_sample)

# Save the model
joblib.dump(calibrated_model, "calibrated_model.pkl")
print("✅ calibrated_model.pkl saved successfully!")