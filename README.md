# 📧 Email Spam Detection using Machine Learning

This project is a **Machine Learning–based Email Spam Detection system** that classifies emails as **Spam** or **Not Spam (Ham)** using **Natural Language Processing (NLP)** techniques and multiple ML models.  
It also includes a **Streamlit web app** for real-time predictions.

---

## 🚀 Features
- Text preprocessing using **NLTK**
- Feature extraction with **TF-IDF**
- Multiple Machine Learning models
- Interactive **Streamlit web application**
- Clean and modular code structure
- Beginner-friendly and resume-ready project

---

## 🧠 Machine Learning Models Used
- Naive Bayes  
- Logistic Regression  
- Linear Support Vector Machine (SVM)  

> ⚠️ Note: Linear SVM performs best for text classification with TF-IDF features.

---

## 📂 Dataset
- **SMS Spam Collection Dataset**
- Labels:
  - `spam` → Spam email
  - `ham` → Not spam email

---

## 🛠️ Tech Stack
- Python  
- Pandas  
- Scikit-learn  
- NLTK  
- Streamlit  

---

## 📁 Project Structure
email-spam-detection/
│
├── app.py # Streamlit app
├── data_preprocessing.py # Text cleaning & preprocessing
├── model_training.py # ML model training
├── spam.csv # Dataset
├── requirements.txt # Dependencies
└── README.md

yaml
Copy code

---

## ▶️ How to Run the Project

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
2️⃣ Run Streamlit app
bash
Copy code
streamlit run app.py
3️⃣ Open in browser
arduino
Copy code
http://localhost:8501
🧪 Example Test Emails
🚨 Spam Example
css
Copy code
URGENT!!! Your bank account has been suspended.
Verify your details immediately to avoid closure.
Click here now.
✅ Not Spam Example
swift
Copy code
Hi Aadarsh,

Our meeting is scheduled for tomorrow at 11 AM.
Please let me know if you need any changes.

Thanks,
Rahul
📊 Output
User enters email text

Selects ML model

App predicts:

🚨 SPAM

✅ NOT SPAM

📌 Future Improvements
Add confidence score

Model persistence (save & load)

Deep Learning models (LSTM / BERT)

Online deployment (Streamlit Cloud)

Explainable AI (feature importance)

👨‍💻 Author
Aadarsh Shukla
Aspiring Data Scientist & ML Engineer
YouTuber | Freelancer | Python & ML Enthusiast

⭐ If you like this project
Give it a ⭐ on GitHub — it motivates me to build more ML projects!
