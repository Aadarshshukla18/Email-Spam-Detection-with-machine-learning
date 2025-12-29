import streamlit as st
from data_preprocessing import load_and_clean_data, preprocess_text
from model_training import train_models
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Email Spam Detection", layout="centered")

st.title("📧 Email Spam Detection App")
st.write("Detect whether an email message is **Spam** or **Not Spam (Ham)**")

@st.cache_resource
def load_model():
    df = load_and_clean_data("spam.csv")

    X = df["message"]
    y = df["label"]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = train_models(X_train, y_train)
    return models

models = load_model()

user_input = st.text_area("✉️ Enter email text here:")

model_choice = st.selectbox(
    "Choose ML Model",
    list(models.keys()),
    index=list(models.keys()).index("Linear SVM")
)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        model = models[model_choice]
        prediction = model.predict([user_input])[0]

        if prediction == 1:
            st.error("🚨 This email is **SPAM**")
        else:
            st.success("✅ This email is **NOT SPAM**")
