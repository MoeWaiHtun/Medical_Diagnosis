import pandas as pd
import os

def load_all_datasets(lang: str = "en"):
    """
    Language ('en' သို့မဟုတ် 'my') အပေါ် မူတည်၍ data_en သို့မဟုတ် data_my Folder မှ
    Dataset ဖိုင်များကို Dynamic ဖတ်ယူပြီး Dictionary များအဖြစ် ပြောင်းလဲပေးသည့် Function
    """
    # ဘာသာစကားအလိုက် Folder လမ်းကြောင်း သတ်မှတ်ခြင်း (ဥပမာ- data_en သို့မဟုတ် data_my)
    data_dir = f"data_{lang}" if os.path.exists(f"data_{lang}") else "data"

    dataset_path = os.path.join(data_dir, "dataset.csv")
    desc_path = os.path.join(data_dir, "system_Description.csv")
    precaution_path = os.path.join(data_dir, "system_precaution.csv")
    severity_path = os.path.join(data_dir, "Symptom-severity.csv")

    df_main = pd.read_csv(dataset_path)
    df_desc = pd.read_csv(desc_path)
    df_prec = pd.read_csv(precaution_path)
    df_sev = pd.read_csv(severity_path)

    # 1. Severity Dict (မြန်မာ / အင်္ဂလိပ် Column အမည် နှိုင်းယှဉ်၍ ယူခြင်း)
    if 'ရောဂါလက္ခဏာ (မြန်မာ)' in df_sev.columns:
        symptom_col = 'ရောဂါလက္ခဏာ (မြန်မာ)'
        weight_col = 'ပြင်းထန်မှုအဆင့်'
    else:
        symptom_col = 'Symptom'
        weight_col = 'weight'

    df_sev[symptom_col] = df_sev[symptom_col].astype(str).str.strip().str.lower().str.replace('_', ' ')
    severity_dict = dict(zip(df_sev[symptom_col], df_sev[weight_col]))

    # 2. Description Dict (မြန်မာ / အင်္ဂလိပ် Column အမည် နှိုင်းယှဉ်၍ ယူခြင်း)
    disease_col = 'ရောဂါအမည်' if 'ရောဂါအမည်' in df_desc.columns else 'Disease'
    desc_col = 'ရောဂါဖော်ပြချက်' if 'ရောဂါဖော်ပြချက်' in df_desc.columns else 'Description'

    df_desc[disease_col] = df_desc[disease_col].astype(str).str.strip()
    desc_dict = dict(zip(df_desc[disease_col], df_desc[desc_col]))

    # 3. Precaution Dict
    disease_prec_col = 'ရောဂါအမည်' if 'ရောဂါအမည်' in df_prec.columns else 'Disease'
    df_prec[disease_prec_col] = df_prec[disease_prec_col].astype(str).str.strip()
    prec_dict = {}
    
    # Precaution Column များကို dynamic ရှာဖွေခြင်း (ကြိုတင်ကာကွယ်ရန်_ သို့မဟုတ် Precaution_)
    prec_cols = [c for c in df_prec.columns if c.startswith('ကြိုတင်ကာကွယ်ရန်') or c.startswith('Precaution')]
    
    for idx, row in df_prec.iterrows():
        precautions = [row[col] for col in prec_cols if pd.notna(row[col]) and str(row[col]).strip() != '']
        prec_dict[row[disease_prec_col]] = precautions

    return df_main, desc_dict, prec_dict, severity_dict