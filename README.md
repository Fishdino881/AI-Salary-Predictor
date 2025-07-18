
AI Salary Predictor

AI Salary Predictor is a machine learning-powered web application built using Streamlit and Scikit-learn. It provides accurate salary estimations based on various professional factors like experience, education level, job role, certifications, skills, and performance rating. This tool helps users understand their market value and make data-driven career decisions.

---

 Features

- 🔍 Real-time Salary Prediction
- 📊 Skill Impact Visualization
- 📈 Industry Salary Trends
- 🏆 Achievements Dashboard
- 📁 PDF Report Export
- 🎨 Custom UI with CSS and Animations

---

Tech Stack

- Python
- Streamlit
- Scikit-learn
- Pandas, NumPy
- GridSearchCV (for model tuning)
- Joblib & Pickle (for model persistence)
- Custom CSS for animations and style

---

Project Structure

```
├── app.py                # Main Streamlit application
├── train_model.py        # Model training pipeline with hyperparameter tuning
├── style.css             # Custom styles for UI
├── artifacts/            # Stores trained models and metadata
└── data/
    └── salary_data.csv   # Dataset used for training
```

---

Run Locally

1. Clone the repo  
```bash
git clone https://github.com/yourusername/ai-salary-predictor.git
cd ai-salary-predictor
```

2. Install dependencies  
```bash
pip install -r requirements.txt
```

3. Train the model (optional)  
```bash
python train_model.py
```

4. Run the app
```bash
streamlit run app.py
```

---

Example Inputs

- Job Role: Data Scientist
- Education Level: Master's
- Years of Experience: 5
- Skills: Python, Machine Learning
- Certifications: 2
- Remote Work: Yes
- Management Role: No
- Performance Rating: Above Average

---

Author

Developed with ❤️ by [Kunal Singh](https://www.linkedin.com/in/kunal-singh-699485215)
