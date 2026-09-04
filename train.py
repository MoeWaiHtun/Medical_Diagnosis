import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# AI Machine Learning Algorithms
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from utils.data_loader import load_all_datasets

def train_and_save_models():
    print("⏳ Dataset များကို စတင်ဖတ်ယူနေပါသည်...")
    
    # ၁။ Data များ ဖတ်ယူခြင်း
    df_dataset, desc_dict, prec_dict, severity_dict = load_all_datasets()
    
    # 'ရောဂါလက္ခဏာ' ဖြင့်စသော Column များကို ရှာဖွေပြီး စာကြောင်းအဖြစ် ပေါင်းစပ်ခြင်း
    symptom_cols = [c for c in df_dataset.columns if c.startswith('ရောဂါလက္ခဏာ')]
    
    df_dataset['symptom_text'] = df_dataset[symptom_cols].apply(
        lambda row: ' '.join(row.dropna().astype(str).str.strip()),
        axis=1
    )
    
    X_text = df_dataset['symptom_text']
    y_raw = df_dataset['ရောဂါအမည်'].astype(str).str.strip()
    
    # ၂။ Label Encoding (ရောဂါအမည်များကို ကိန်းဂဏန်းပြောင်းခြင်း)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    # Train / Test Split (80% Train, 20% Test)
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # ၃။ TF-IDF Vectorization
    print("⏳ TF-IDF Feature Extraction ပြုလုပ်နေပါသည်...")
    tfidf = TfidfVectorizer(max_features=150, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train_text)
    X_test_tfidf = tfidf.transform(X_test_text)
    
    # ၄။ AI Algorithms မော်ဒယ်များ သတ်မှတ်ခြင်းနှင့် Hyperparameter Tuning
    models_config = {
        "Random Forest": (
            RandomForestClassifier(random_state=42),
            {"n_estimators": [50, 100], "max_depth": [None, 10, 20]}
        ),
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, random_state=42),
            {"C": [0.1, 1.0, 10.0]}
        ),
        "SVM": (
            SVC(probability=True, random_state=42),
            {"C": [0.1, 1.0, 10.0], "kernel": ["linear", "rbf"]}
        ),
        "Naive Bayes": (
            MultinomialNB(),
            {"alpha": [0.1, 0.5, 1.0]}
        ),
        "MLP Classifier": (
            MLPClassifier(max_iter=500, random_state=42),
            {"hidden_layer_sizes": [(50,), (100,)], "activation": ["relu", "tanh"]}
        ),
        "XGBoost": (
            XGBClassifier(eval_metric='mlogloss', random_state=42),
            {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]}
        )
    }
    
    trained_models = {}
    model_results = []
    
    print("\n🚀 AI မော်ဒယ်များကို စတင်လေ့ကျင့်ပေးနေပါသည်...")
    
    for name, (model, param_grid) in models_config.items():
        print(f"🔄 Training {name}...")
        grid = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train_tfidf, y_train)
        
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test_tfidf)
        acc = accuracy_score(y_test, y_pred)
        
        trained_models[name] = best_model
        model_results.append({"Model": name, "Accuracy": acc})
        print(f"✅ {name} သင်ကြားပြီးပါပြီ။ Accuracy: {acc * 100:.2f}%")
        
    # ၅။ Artifacts များကို models/ ဖိုဒါထဲသို့ သိမ်းဆည်းခြင်း
    os.makedirs("models", exist_ok=True)
    
    print("\n💾 Model Artifacts များကို သိမ်းဆည်းနေပါသည်...")
    joblib.dump(trained_models, "models/all_models.pkl")
    joblib.dump(tfidf, "models/tfidf.pkl")
    joblib.dump(le, "models/le.pkl")
    joblib.dump(model_results, "models/model_results.pkl")
    joblib.dump(X_train_tfidf, "models/X_train_tfidf.pkl")
    joblib.dump(y_train, "models/y_train.pkl")
    joblib.dump(le.classes_, "models/classes.pkl")
    
    print("\n🎉 မော်ဒယ်များ သင်ကြားသိမ်းဆည်းခြင်း အောင်မြင်စွာ ပြီးဆုံးပါပြီ။")

if __name__ == "__main__":
    train_and_save_models()