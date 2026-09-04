import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys

def generate_comparison_report(lang: str = "en"):
    # ၁။ ဘာသာစကားအလိုက် Data နှင့် Model Folder လမ်းကြောင်းများ သတ်မှတ်ခြင်း
    data_dir = f"data_{lang}" if os.path.exists(f"data_{lang}") else "data"
    model_dir = f"models/{lang}" if os.path.exists(f"models/{lang}") else "models"
    
    # Dataset ကို ဖတ်ယူခြင်း
    df = pd.read_csv(os.path.join(data_dir, "dataset.csv"))
    
    # ၂။ 'ရောဂါလက္ခဏာ' သို့မဟုတ် 'Symptom' ဖြင့် စသော Column များကို Dynamic ရှာဖွေခြင်း
    symptom_prefix = 'ရောဂါလက္ခဏာ' if any(c.startswith('ရောဂါလက္ခဏာ') for c in df.columns) else 'Symptom'
    symptom_cols = [c for c in df.columns if c.startswith(symptom_prefix)]
    
    # ရောဂါလက္ခဏာများကို စာကြောင်းအဖြစ် ပေါင်းစည်းခြင်း
    df['symptom_text'] = df[symptom_cols].apply(
        lambda row: ' '.join(row.dropna().astype(str).str.strip()),
        axis=1
    )
    
    X_text = df['symptom_text']
    
    # ၃။ 'ရောဂါအမည်' သို့မဟုတ် 'Disease' Column ကို Target Label အဖြစ် သတ်မှတ်ခြင်း
    target_col = 'ရောဂါအမည်' if 'ရောဂါအမည်' in df.columns else 'Disease'
    y = df[target_col].astype(str).str.strip()
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    
    # ၄။ သက်ဆိုင်ရာ ဘာသာစကား Model Folder ထဲမှ Model များကို Load လုပ်ခြင်း
    tfidf = joblib.load(os.path.join(model_dir, "tfidf.pkl"))
    X_test_tfidf = tfidf.transform(X_test_text)
    
    all_models = joblib.load(os.path.join(model_dir, "all_models.pkl"))
    le = joblib.load(os.path.join(model_dir, "le.pkl"))
    
    for name, model in all_models.items():
        y_pred = model.predict(X_test_tfidf)
        print(f"\n===== {name} [{lang.upper()}] =====")
        print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))
        
        # Confusion matrix heatmap ရေးဆွဲခြင်း
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(12,10))
        
        # Matplotlib/Seaborn တွင် မြန်မာစာလုံးများ မှန်ကန်စွာ ပေါ်စေရန် Font ပြင်ဆင်ခြင်း
        plt.rcParams['font.sans-serif'] = ['Padauk', 'Pyidaungsu', 'Arial']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title(f'Confusion Matrix - {name} ({lang.upper()})')
        
        # ဘာသာစကားအလိုက် X, Y label များ ပြောင်းလဲခြင်း
        if lang == "my":
            plt.xlabel('Predicted (ခန့်မှန်းရလဒ်)')
            plt.ylabel('Actual (မူလရောဂါ)')
        else:
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            
        plt.tight_layout()
        
        # Output ပုံများကို သက်ဆိုင်ရာ Model Folder ထဲတွင် သိမ်းဆည်းခြင်း
        plt.savefig(os.path.join(model_dir, f'cm_{name}.png'))
        plt.close()

if __name__ == "__main__":
    # Terminal မှ run သောအခါ ဘာသာစကား ထည့်သွင်းနိုင်ရန် ပြင်ဆင်ထားသည် 
    # ဥပမာ - `python compare_models.py my` သို့မဟုတ် `python compare_models.py en`
    lang_arg = sys.argv[1] if len(sys.argv) > 1 else "en"
    generate_comparison_report(lang=lang_arg)