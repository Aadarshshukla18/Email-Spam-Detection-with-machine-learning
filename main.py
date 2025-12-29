from data_preprocessing import load_and_clean_data, preprocess_text
from model_training import train_models
from model_evaluation import evaluate_model
from sklearn.model_selection import train_test_split


# Load data
data_path = "spam.csv"
df = load_and_clean_data(data_path)

# Preprocess messages
df['message'] = df['message'].apply(preprocess_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

# Train models
models = train_models(X_train, y_train)

# Evaluate models
for model_name, model in models.items():
    evaluate_model(model, X_test, y_test, model_name)

# Test custom input
sample_email = ["Congratulations! You have won a free lottery ticket."]
for model_name, model in models.items():
    result = model.predict(sample_email)
    print(f"{model_name} Prediction:", "Spam" if result[0] == 1 else "Not Spam")
