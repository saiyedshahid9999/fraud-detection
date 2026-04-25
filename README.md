# 💳 Credit Card Fraud Detection System

## 📌 Overview
This project is an end-to-end Machine Learning system for detecting fraudulent credit card transactions using real-world datasets.

It focuses on identifying rare fraud cases in highly imbalanced data and provides a real-time prediction API using FastAPI.

---

## 🚀 Features
- Fraud detection using XGBoost model
- Handles highly imbalanced dataset (~0.17% fraud cases)
- Real-time prediction API using FastAPI
- Scalable and modular structure
- Risk scoring system (Low / Medium / High)

---

## 🛠️ Tech Stack
- Python
- FastAPI
- XGBoost
- Pandas, NumPy, Scikit-learn
- REST API

---

## 📂 Project Structure
fraud-detection/
│
├── main.py # FastAPI application
├── train_model.py # Model training script
├── xgb_fraud_model.pkl # Trained model
├── scaler.pkl # Scaler
├── notebooks/ # Experiments
├── src/ # Core logic
├── README.md


---

## ⚠️ Dataset
Dataset is not included due to size limitations.

Download here:
👉 https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place it in:

data/raw/creditcard.csv


---

## ▶️ How to Run

### 1. Clone the repository

git clone https://github.com/saiyedshahid9999/fraud-detection.git
cd fraud-detection
2. Install dependencies
pip install -r requirements.txt
3. Train the model
python train_model.py
4. Run API
uvicorn main:app --reload --port 8001
5. Open Swagger UI
http://127.0.0.1:8001/docs
📊 API Demo
🔹 Swagger Interface

🔹 Input Example

🔹 Prediction Result (High Risk)

📈 Example Output
{
  "fraud_probability": 1,
  "risk_score": 100,
  "decision": "HIGH RISK"
}
📊 Problem Solved

Credit card fraud detection is challenging due to:

Extreme class imbalance
Need for real-time decisions

This system improves detection accuracy while minimizing false positives.

💡 Future Improvements
Deploy on AWS / Azure
Add real-time streaming (Kafka)
Improve model using deep learning
👨‍💻 Author

Shahid Saiyed
MSc AI for Business (London)


---

# 🔥 Small but IMPORTANT fix

👉 Your image names have spaces  
GitHub needs `%20`

You already did correctly:

Screenshot%202026-04-25%20202234.png


✔ Good — don’t change that

---

# 🚀 Final Git push (after editing README)


git add .
git commit -m "Updated README with API screenshots"
git push
