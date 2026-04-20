# 🩺 Hypertension Prediction (Streamlit App)

A Machine Learning web app to predict hypertension risk using patient health data.

---

## 🚀 Features

* Real-time prediction using Streamlit
* XGBoost-based ML model
* Optimized for **high recall (~90%)**
* Simple and interactive UI

---

## 🧠 Model Info

* Algorithm: XGBoost
* Hyperparameter tuning: Optuna
* Focus: **Don't miss high-risk patients**

---

## 📂 Project Structure

├── app.py
├── model.pkl
├── scaler.pkl
├── requirements.txt
├── README.md

---

## ⚙️ Setup

```bash
git clone https://github.com/your-username/hypertension-app.git
cd hypertension-app
pip install -r requirements.txt
```

---

## ▶️ Run

```bash
streamlit run app.py
```

---

## ⚠️ Important Note

This model prioritizes **recall over precision**
→ It may give some false positives
→ But avoids missing real patients

---

## 📌 Future Work

* Add SHAP explainability
* Improve UI
* Deploy on cloud

---

## 📜 License

MIT License
