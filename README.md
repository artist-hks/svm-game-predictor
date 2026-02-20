# 🎮 Video Game Sales Class Predictor (SVM)

An end-to-end Machine Learning web application that predicts whether a video game will achieve **Low**, **Medium**, or **High** global sales based on regional sales data.

This project demonstrates the complete ML pipeline from data preprocessing and model training to deployment using Streamlit.

---

## 🚀 Live Demo

👉 *(https://svm-game-predictor-5u4hudbxahhnub9u9s3eqb.streamlit.app/)*

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


---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
git clone https://github.com/artist-hks/svm-game-sales-predictor.git
cd svm-game-sales-predictor

### 2️⃣ Install dependencies
python -m pip install -r requirements.txt

### 3️⃣ Train the model (optional)
python train_model.py

### 4️⃣ Run the Streamlit app
python -m streamlit run app.py

---

## 🎯 How the Predictor Works
The user provides regional sales values:

-NA Sales
-EU Sales
-JP Sales
-Other Sales

The trained SVM model predicts the sales category:

-📉 Low Sales
-📊 Medium Sales
-🚀 High Sales

## 🛠️ Tech Stack
-Python
-Scikit-learn
-Pandas
-NumPy
-Streamlit
-Joblib
-Matplotlib
-Seaborn


## 👨‍💻 Author
Hemant Sharma (HKS)
Computer Science Student, PIET Jaipur
-🎨 UI/UX Designer
-💻 Web Developer
-🤖 Machine Learning Enthusiast
-🎮 Game Design & Development


## ⭐ Future Improvements
-Streamlit UI enhancement
-Model comparison dashboard
-Advanced feature engineering
-Automated cloud deployment
-Real-time game analytics integration


