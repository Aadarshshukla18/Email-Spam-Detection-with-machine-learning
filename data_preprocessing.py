import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

def load_and_clean_data(data_path="spam.csv"):
    df = pd.read_csv(data_path, encoding="latin-1")

    # Keep only required columns
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']

    # Convert labels to binary
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    return df

    
    # Keep only required columns
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
    
    # Convert labels to binary
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    
    return df

def preprocess_text(text):
    ps = PorterStemmer()
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [ps.stem(word) for word in text if word not in stopwords.words('english')]
    return ' '.join(text)
