import re
import pandas as pd
from sklearn.model_selection import train_test_split

def clean_text(text: str) -> str:
    text = re.sub(r'<[^>]+>', '', text)      
    text = re.sub(r'\s+', ' ', text)         
    return text.strip()

def load_and_preprocess(path: str):
    df = pd.read_csv(path)
    
    df['review'] = df['review'].apply(clean_text)
    df['label']  = df['sentiment'].map({'positive': 1, 'negative': 0})
    df = df.drop(columns=['sentiment'])
    
    train, temp = train_test_split(df, test_size=0.2,  random_state=42, stratify=df['label'])
    val,   test = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp['label'])
    
    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    return train, val, test

if __name__ == "__main__":
    train, val, test = load_and_preprocess("data/raw/archive(3).zip")
    
    train.to_csv("data/processed/train.csv", index=False)
    val.to_csv("data/processed/val.csv",   index=False)
    test.to_csv("data/processed/test.csv", index=False)
    
    print(" Saved to data/processed/")