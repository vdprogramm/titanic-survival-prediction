import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from pathlib import Path

DATA_PATH = Path("data/train.csv")
MODEL_PATH = Path("models/titanic_model.pkl")

def prepare_data(df, age_median, fare_median, embarked_mode):
    df = df.copy()
    # Fill missing values
    df["Age"] = df["Age"].fillna(age_median)
    df["Fare"] = df["Fare"].fillna(fare_median)
    df["Embarked"] = df["Embarked"].fillna(embarked_mode)
    
    # Convert categorical to numeric
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)
    
    # Select features
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"] + \
               [c for c in df.columns if c.startswith("Embarked_")]
    return df[features]

def main():
    df = pd.read_csv(DATA_PATH)
    
    # Calculate median/mode for imputation
    age_median = df["Age"].median()
    fare_median = df["Fare"].median()
    embarked_mode = df["Embarked"].mode()[0]
    
    X = prepare_data(df, age_median, fare_median, embarked_mode)
    y = df["Survived"]
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    print(f"Validation Accuracy: {accuracy_score(y_val, predictions):.4f}")
    
    # Train final model using all labeled data
    model.fit(X, y)
    
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "columns": X.columns.tolist(),
        "age_median": age_median,
        "fare_median": fare_median,
        "embarked_mode": embarked_mode
    }, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
