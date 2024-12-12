import streamlit as st
import numpy as np
import joblib

class LogisticRegression:
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.n_iters):
            model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(model)
            dw = (1 / num_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / num_samples) * np.sum(y_pred - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        predictions = []
        for p in y_pred:
            if p >= 0.5:
                predictions.append(1)
            else:
                predictions.append(0)

        return np.array(predictions)

    def score(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy


# Custom CSS for background color and text styling
st.markdown(
    """
    <style>
    body {
        background-color: #deeee;
        color: #31333F;
    }
    .main {
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }
    h1, h2, h3 {
        color: white;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown('<div class="main">', unsafe_allow_html=True)
st.title("Stroke Risk Prediction")

# Collect Input Features
st.header("Enter the values")

# Input Fields
age = st.number_input("Age", min_value=0, max_value=120, value=0)
hypertension = st.selectbox("Hypertension (0: No, 1: Yes)", [0, 1])
gender = st.selectbox("Gender", ["Female", "Male"])
heart_disease = st.selectbox("Heart Disease (0: No, 1: Yes)", [0, 1])
bmi = st.number_input("BMI", min_value=0.0, value=20.0)
avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, value=20.0)
residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "Children", "Never_worked"])
smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])
ever_married = st.selectbox("Ever Married (0: No, 1: Yes)", [0, 1])

# Encode categorical inputs to numerical values
gender_encoded = 1 if gender == "Female" else 0
residence_encoded = 1 if residence_type == "Urban" else 0
work_type_encoded = {"Private": 0, "Self-employed": 1, "Govt_job": 2, "Children": 3, "Never_worked": 4}[work_type]
smoking_status_encoded = {"formerly smoked": 0, "never smoked": 1, "smokes": 2, "Unknown": 3}[smoking_status]

# Prepare the input features (Ensure there are 17 features, filling missing ones with zeros)
input_features = np.array([[age, gender_encoded, hypertension, heart_disease, avg_glucose_level,
                            bmi, residence_encoded, work_type_encoded, smoking_status_encoded, ever_married,
                            0, 0, 0, 0, 0, 0, 0]])  # Add any missing features as 0 or appropriate values

# Load the trained model using joblib
@st.cache_resource  # Cache the model to avoid reloading on every interaction
def load_model():
    return joblib.load("model.sav")

model = load_model()

# Predict Button
if st.button("Predict Stroke Risk"):
    # Predict the probability
    y_pred = model.predict(input_features)  # Use the 'predict' method

    # If your Logistic Regression is built as a custom implementation, it will output either 0 or 1
    # So, you might want to use the model's predicted probabilities instead:
    probability = model.sigmoid(np.dot(input_features, model.weights) + model.bias) * 100  # For custom sigmoid output

    # Display Stroke Probability
    st.write(f"### Probability: {probability[0]:.2f}%")

    # Display Risk Level
    if probability >= 50:
        st.error("High risk")
    else:
        st.success("Low risk")

st.markdown('</div>', unsafe_allow_html=True)
