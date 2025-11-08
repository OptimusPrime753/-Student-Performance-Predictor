# 🎓 Student Performance Predictor (Streamlit + Machine Learning)

This Streamlit web app predicts a student's academic performance (Low/Middle/High) based on their study hours, attendance, health, and extracurricular activity levels.  
It demonstrates how **machine learning** can be integrated into an interactive **Streamlit UI** for real-time prediction and data visualization.

---

## 🚀 Objective

To build an **end-to-end machine learning application** that:
- Collects student-related data through a simple UI.
- Predicts academic performance using ML models.
- Demonstrates interactive data visualization.
- Helps explore how AI-based tools can support education analytics.

---

## 🧠 Model Overview

Two models are trained on a **synthetic dataset** generated with realistic student data patterns:

| Model | Algorithm | Purpose | Accuracy |
|--------|------------|----------|-----------|
| `Decision Tree Classifier` | Supervised ML | Interpretable model for simple decision-making | ~75% |
| `Logistic Regression` | Linear ML | Robust probabilistic classifier | ~82% |

### 🎯 Features Used for Prediction:
- **Study Hours**: Number of hours studied per day.
- **Attendance %**: Percentage of class attendance.
- **Health**: Rated on a scale of 1–10.
- **Extracurricular Activity Level**: Rated on a scale of 1–10.
- **Age**, **Gender**, **Parent Education Level**, **Internet Access**, **Sleep Hours** (used internally in the dataset).

---

## 🧩 How It Works

1. The app **generates a synthetic training dataset** on the first load.  
2. Two ML models (Decision Tree & Logistic Regression) are **trained automatically**.  
3. Users can input custom values through the sidebar.  
4. The app encodes categorical features and makes predictions using the selected model.  
5. Real-time visualisations (accuracy, feature importances) are displayed interactively.

---

## 🖥️ Running the App Locally

Download the file and run it locally with terminal
Use the command streamlit run "Codefile"

### 1️⃣ Clone or Download the Repository
```bash
git clone https://github.com/your-username/student-performance-predictor.git
cd Student performance
