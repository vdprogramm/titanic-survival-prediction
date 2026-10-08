import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier # Imported to avoid ModuleNotFoundError

MODEL_PATH = Path(__file__).resolve().parent / "models" / "titanic_model.pkl"

st.set_page_config(page_title="Titanic Survival Prediction", page_icon="🚢")
st.title("🚢 Titanic Survival Prediction V2")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

bundle = load_model()

if bundle is None:
    st.error("Model not found. Please train the model first.")
    st.stop()

model = bundle["model"]
columns = bundle["columns"]
age_median = bundle.get("age_median", 28.0)
fare_median = bundle.get("fare_median", 14.45)

st.success("Model loaded successfully!")
st.write("Enter passenger details to predict survival.")

with st.form("passenger_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
        sex = st.selectbox("Sex", ["male", "female"])
        age = st.number_input("Age", min_value=0.0, max_value=120.0, value=float(age_median))
        fare = st.number_input("Fare", min_value=0.0, max_value=1000.0, value=float(fare_median))
        
    with col2:
        sibsp = st.number_input("Number of Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10, value=0)
        parch = st.number_input("Number of Parents/Children Aboard (Parch)", min_value=0, max_value=10, value=0)
        embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])
        title = st.selectbox("Title", ["Mr", "Mrs", "Miss", "Master", "Rare"])
        
    submitted = st.form_submit_button("Predict Survival")

if submitted:
    family_size = sibsp + parch + 1
    passenger = {
        "Pclass": pclass,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "FamilySize": family_size,
        "IsAlone": int(family_size == 1),
        "Sex_male": sex == "male",
        "Sex": 0 if sex == "male" else 1,
        "Embarked_Q": embarked == "Q",
        "Embarked_S": embarked == "S",
        "Title_Miss": title == "Miss",
        "Title_Mr": title == "Mr",
        "Title_Mrs": title == "Mrs",
        "Title_Rare": title == "Rare"
    }
    
    X_input = pd.DataFrame([passenger])
    X_input = X_input.reindex(columns=columns, fill_value=0)
    
    prediction = model.predict(X_input)[0]
    probability = model.predict_proba(X_input)[0]
    survival_index = list(model.classes_).index(1)
    survival_probability = probability[survival_index]
    
    st.divider()
    st.subheader("Prediction Result")
    if prediction == 1:
        st.success("Predicted: Survived")
    else:
        st.error("Predicted: Did Not Survive")
        
    st.metric(
        "Estimated Survival Probability",
        f"{survival_probability * 100:.2f}%"
    )
    st.progress(float(survival_probability))
    st.caption("This is a statistical model prediction, not a historical fact or certainty.")
