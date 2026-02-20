# 🎮 Video Game Sales Class Predictor (SVM)

An end-to-end Machine Learning web application that predicts whether a video game will achieve **Low**, **Medium**, or **High** global sales based on regional sales data.

This project demonstrates the complete ML pipeline from data preprocessing and model training to deployment using Streamlit.

---

## 🚀 Live Demo

👉 *(Add your Streamlit link here after deployment)*

---

## 📌 Project Overview

This project uses a **Support Vector Machine (SVM)** classifier to analyze video game regional sales and predict the overall sales category.

The workflow includes:

- Data preprocessing  
- Feature scaling  
- Hyperparameter tuning (GridSearchCV)  
- Model evaluation  
- Model serialization  
- Interactive web deployment  

---

## 🧠 Machine Learning Pipeline

### 🔹 Data Preparation
- Removed missing values  
- Created sales categories using quantiles  
- Selected key regional sales features  

### 🔹 Feature Engineering
- StandardScaler for normalization  
- Train–test split (70–30)

### 🔹 Model Training
- Support Vector Machine (SVC)  
- Hyperparameter tuning with GridSearchCV  
- Kernel optimization

### 🔹 Evaluation Metrics
- Accuracy score  
- Classification report  
- ROC analysis  
- Precision–Recall analysis  

### 🔹 Deployment
- Model saved using Joblib  
- Streamlit interactive UI  
- Real-time prediction

---

## 📂 Repository Structure
svm-game-sales-predictor
│
├── app.py
├── train_model.py
├── svm_model.pkl
├── scaler.pkl
├── requirements.txt

