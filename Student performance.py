"""
Student Performance Predictor - Complete Standalone Application
A single-file ML application for predicting student performance (Low/Middle/High)
using Decision Tree and Logistic Regression models.

Author: AI Assistant (with Eddy)
Date: 2025
"""

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ==================== ML MODEL CLASS ====================

class StudentPerformancePredictor:
    """
    Machine Learning model for predicting student performance.
    Supports Decision Tree and Logistic Regression algorithms.
    """

    def __init__(self):
        self.decision_tree_model = None
        self.logistic_model = None
        self.label_encoders = {}
        self.target_encoder = None
        self.dt_accuracy = 0
        self.lr_accuracy = 0

    def create_training_data(self, n_samples=480):
        """
        Create synthetic training data based on student features
        """
        np.random.seed(42)
        data = {
            'gender': np.random.choice(['M', 'F'], n_samples),
            'StageID': np.random.choice(['lowerlevel', 'MiddleSchool', 'HighSchool'], n_samples),
            'Relation': np.random.choice(['Father', 'Mum'], n_samples),
            'raisedhands': np.random.randint(0, 101, n_samples),
            'VisITedResources': np.random.randint(0, 101, n_samples),
            'AnnouncementsView': np.random.randint(0, 101, n_samples),
            'Discussion': np.random.randint(0, 101, n_samples),
            'ParentschoolSatisfaction': np.random.choice(['Good', 'Bad'], n_samples),
            'StudentAbsenceDays': np.random.choice(['Under-7', 'Above-7'], n_samples)
        }

        df = pd.DataFrame(data)
        df['avg_engagement'] = (
            df['raisedhands'] +
            df['VisITedResources'] +
            df['AnnouncementsView'] +
            df['Discussion']
        ) / 4

        def assign_class(row):
            score = row['avg_engagement']
            if row['ParentschoolSatisfaction'] == 'Good':
                score += 5
            if row['StudentAbsenceDays'] == 'Under-7':
                score += 5
            if score >= 60:
                return 'H'
            elif score >= 35:
                return 'M'
            else:
                return 'L'

        df['Class'] = df.apply(assign_class, axis=1)
        df = df.drop('avg_engagement', axis=1)
        return df

    def encode_features(self, df, fit=True):
        """
        Encode categorical features to numerical values
        """
        categorical_columns = [
            'gender',
            'StageID',
            'Relation',
            'ParentschoolSatisfaction',
            'StudentAbsenceDays'
        ]
        df_encoded = df.copy()
        for col in categorical_columns:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df_encoded[col] = self.label_encoders[col].fit_transform(df[col])
            else:
                df_encoded[col] = self.label_encoders[col].transform(df[col])
        return df_encoded

    def train_models(self):
        """
        Train both Decision Tree and Logistic Regression models
        """
        df = self.create_training_data()
        X = df.drop('Class', axis=1)
        y = df['Class']
        X_encoded = self.encode_features(X, fit=True)
        self.target_encoder = LabelEncoder()
        y_encoded = self.target_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y_encoded, test_size=0.2, random_state=42
        )

        # Decision Tree
        self.decision_tree_model = DecisionTreeClassifier(random_state=42, max_depth=10)
        self.decision_tree_model.fit(X_train, y_train)
        self.dt_accuracy = self.decision_tree_model.score(X_test, y_test)

        # Logistic Regression
        self.logistic_model = LogisticRegression(random_state=42, max_iter=1000)
        self.logistic_model.fit(X_train, y_train)
        self.lr_accuracy = self.logistic_model.score(X_test, y_test)

        return self.dt_accuracy, self.lr_accuracy

    def predict(self, features, model_type='logistic_regression'):
        """
        Make prediction using specified model
        """
        df = pd.DataFrame([features])
        df_encoded = self.encode_features(df, fit=False)

        if model_type == 'decision_tree':
            model = self.decision_tree_model
            model_name = 'Decision Tree'
            accuracy = f"{self.dt_accuracy * 100:.1f}%"
        else:
            model = self.logistic_model
            model_name = 'Logistic Regression'
            accuracy = f"{self.lr_accuracy * 100:.1f}%"

        pred_encoded = model.predict(df_encoded)[0]
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df_encoded)[0]
            confidence = max(proba) * 100
        else:
            confidence = 100.0

        prediction = self.target_encoder.inverse_transform([pred_encoded])[0]
        category_map = {'H': 'High', 'M': 'Middle', 'L': 'Low'}
        category = category_map[prediction]

        return {
            'prediction': prediction,
            'category': category,
            'confidence': f"{confidence:.2f}",
            'model': model_name,
            'modelAccuracy': accuracy
        }


# ==================== STREAMLIT UI ====================

def main():
    """Main Streamlit application"""

    st.set_page_config(
        page_title="Student Performance Predictor",
        page_icon="🎓",
        layout="wide"
    )

    # Custom CSS
    st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #f5f5dc 0%, #e8f5e9 100%);
        }
        .stButton>button {
            background: linear-gradient(90deg, #20b2aa 0%, #3cb371 100%);
            color: white; font-weight: bold; border: none;
            padding: 0.75rem 2rem; border-radius: 0.5rem;
            font-size: 1.1rem; width: 100%;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #1a9a92 0%, #2e8b57 100%);
        }
        .prediction-box {
            padding: 2rem; border-radius: 1rem; text-align: center; margin-top: 2rem;
        }
        .high-class { background: #d4edda; border: 3px solid #28a745; }
        .middle-class { background: #fff3cd; border: 3px solid #ffc107; }
        .low-class { background: #f8d7da; border: 3px solid #dc3545; }
    </style>
    """, unsafe_allow_html=True)

    @st.cache_resource
    def load_predictor():
        predictor = StudentPerformancePredictor()
        dt_acc, lr_acc = predictor.train_models()
        return predictor, dt_acc, lr_acc

    with st.spinner("Training machine learning models..."):
        predictor, dt_acc, lr_acc = load_predictor()
    st.success(f"Models trained! Decision Tree: {dt_acc:.1%}, Logistic Regression: {lr_acc:.1%}")

    # Header
    st.markdown("""
    <div style="text-align: center;">
        <h1 style="color: #20b2aa;">🎓 Student Performance Predictor</h1>
        <p style="font-size: 1.2rem; color: #666;">Predict student academic performance using machine learning</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Model Configuration")
        model_type = st.radio(
            "Select Prediction Model",
            options=['logistic_regression', 'decision_tree'],
            format_func=lambda x: "🎯 Logistic Regression" if x == 'logistic_regression' else "🌳 Decision Tree",
            index=0
        )

        st.markdown("---")
        st.markdown("### 📊 About the Models")
        st.info("""
        **Decision Tree:** Interpretable model that splits based on student features.
        **Logistic Regression:** Statistical model using probabilities for classification.
        """)

        st.markdown("---")
        st.markdown("### 📚 Performance Classes")
        st.success("**High (H)**: Excellent performance")
        st.warning("**Middle (M)**: Average performance")
        st.error("**Low (L)**: Needs improvement")

        # Optional download dataset
        if st.button("📥 Download Training Data"):
            df = predictor.create_training_data()
            st.download_button(
                "Download CSV",
                df.to_csv(index=False),
                "training_data.csv",
                "text/csv"
            )

    # Inputs
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("👤 Student Information")
        gender = st.selectbox("Gender", ['M', 'F'], format_func=lambda x: "Male" if x == 'M' else "Female")
        stage = st.selectbox("Education Stage", ['lowerlevel', 'MiddleSchool', 'HighSchool'])
        relation = st.selectbox("Responsible Parent", ['Father', 'Mum'])
        parent_satisfaction = st.selectbox("Parent School Satisfaction", ['Good', 'Bad'])
        absence_days = st.selectbox("Student Absence Days", ['Under-7', 'Above-7'])
    with col2:
        st.subheader("📊 Engagement Metrics")
        raised_hands = st.slider("🙋 Raised Hands", 0, 100, 50)
        visited_resources = st.slider("📚 Visited Resources", 0, 100, 50)
        announcements = st.slider("📢 Announcements Viewed", 0, 100, 50)
        discussion = st.slider("💬 Discussion Participation", 0, 100, 50)

    st.markdown("---")

    if st.button("🔮 Predict Performance", use_container_width=True):
        with st.spinner("🧠 Analyzing student data..."):
            features = {
                'gender': gender,
                'StageID': stage,
                'Relation': relation,
                'raisedhands': raised_hands,
                'VisITedResources': visited_resources,
                'AnnouncementsView': announcements,
                'Discussion': discussion,
                'ParentschoolSatisfaction': parent_satisfaction,
                'StudentAbsenceDays': absence_days
            }

            result = predictor.predict(features, model_type)
            st.toast("✅ Prediction complete!")

            category = result['category']
            if category == 'High':
                class_style, emoji, message = 'high-class', '🌟', "🎉 Excellent! High academic potential!"
            elif category == 'Middle':
                class_style, emoji, message = 'middle-class', '⭐', "👍 Good performance. Some improvements can help."
            else:
                class_style, emoji, message = 'low-class', '💡', "📚 Needs additional academic support."

            st.markdown(f"""
            <div class="prediction-box {class_style}">
                <h1>{emoji} {category} Class</h1>
                <h3>Prediction: {result['prediction']}</h3>
                <p><strong>Confidence:</strong> {result['confidence']}%</p>
                <p><strong>Model:</strong> {result['model']} ({result['modelAccuracy']} Accuracy)</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### 📈 Analysis Metrics")
            colA, colB, colC, colD = st.columns(4)
            avg_engagement = (raised_hands + visited_resources + announcements + discussion) / 4
            with colA: st.metric("Engagement Score", f"{avg_engagement:.1f}")
            with colB: st.metric("Predicted Class", result['prediction'])
            with colC: st.metric("Confidence", f"{result['confidence']}%")
            with colD: st.metric("Model Accuracy", result['modelAccuracy'])

            if category == 'High': st.success(message)
            elif category == 'Middle': st.info(message)
            else: st.warning(message)

            st.markdown("### 🔍 Feature Analysis")
            feature_values = {
                'Raised Hands': raised_hands,
                'Visited Resources': visited_resources,
                'Announcements': announcements,
                'Discussion': discussion
            }
            st.bar_chart(feature_values)

            st.info("""
            **Key Factors for High Performance:**
            - Engagement above 75
            - Good parent satisfaction
            - Low absence days
            - Active participation
            """)

    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p><strong>Powered by Machine Learning</strong></p>
        <p>Decision Tree & Logistic Regression Models</p>
        <p style="font-size: 0.9rem;">Built with Streamlit • scikit-learn • Python</p>
    </div>
    """, unsafe_allow_html=True)


# ==================== RUN APP ====================

if __name__ == "__main__":
    main()
