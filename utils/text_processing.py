import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# NLTK Data များကို ဒေါင်းလုဒ်ဆွဲခြင်း (အသုံးမပြုမီ တစ်ကြိမ်သာ ဒေါင်းရန်လိုအပ်ပါသည်)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

# Lemmatizer Object ဖန်တီးခြင်း
lemmatizer = WordNetLemmatizer()

# ==========================================
# 1. MYANMAR TEXT PROCESSING CONFIGURATION
# ==========================================
MY_NEGATION_PATTERNS = [
    # Pattern 1: စကားလုံး နောက်တွင် မရှိ/မဖြစ် စသည်တို့ တိုက်ရိုက် သို့မဟုတ် တာမျိုး/လည်း ပါပြီး ငြင်းပယ်ခြင်း
    r'[^\s]+\s*(တာမျိုး|တော့|လည်း|တာ|ပါ)?\s*(မရှိ|မဖြစ်)\s*(ဘူး|ပါဘူး|နေဘူး|တော့ဘူး|ပါ|နေတာ)',
    
    # Pattern 2: ရှေ့မှ မ ခံပြီး ငြင်းပယ်ထားသော စကားလုံးများ
    r'မ\s*[^\s]+\s*(ဘူး|ပါဘူး|နေဘူး|ထားဘူး|တာမျိုး|တော့ဘူး)',
    
    # Pattern 3: မရှိသော / မရှိသည့် နောက်တွင် ပါသည့် စကားလုံးများ
    r'မရှိ\s*(သော|သည့်)?\s*[^\s]+',
]

MYANMAR_STOP_WORDS = {
    'ကျွန်တော်', 'ကျနော်', 'ကျွန်တော်တို့', 'ကျနော်တို့', 'ကျွန်မ', 'ကျမ', 'ကျွန်မတို့', 'ကျမတို့',
    'ငါ', 'ငါတို့', 'သူ', 'သူမ', 'သူတို့', 'မိမိ', 'ကိုယ်',
    'သည်', 'မှာ', 'ကို', '၏', 'မှ', 'သို့', 'ဖြင့်', 'နှင့်', 'ဟာ', 'က', 'လည်း', 'အထိ',
    'ရှိ', 'ရှိသည်', 'ရှိတယ်', 'ရှိပါတယ်', 'ရှိနေတယ်',
    'ဖြစ်', 'ဖြစ်သည်', 'ဖြစ်တယ်', 'ဖြစ်ပါတယ်', 'ဖြစ်နေ', 'ဖြစ်နေသည်', 'ဖြစ်နေတယ်',
    'ပါ', 'ပါတယ်', 'ပါသည်', 'တယ်', 'နေတယ်', 'နေသည်', 'နေပါတယ်', 'ရ', 'ရတယ်',
    'အလွန်', 'အင်မတန်', 'ရမ်း', 'အရမ်း', 'နည်းနည်း', 'ခဏခဏ', 'ဒါပေမဲ့', 'ဒါပေမယ့်', 'သို့သော်',
    'ခံစား', 'ခံစားရ', 'ခံစားနေရ', 'ခံစားနေရတယ်', 'ခံစားရတယ်',
    'ပြ', 'ပြသည်', 'ပြနေတယ်', 'ဆရာ', 'ဆရာဝန်', 'ကူညီ', 'ကျေးဇူးပြု၍'
}

# ==========================================
# 2. ENGLISH TEXT PROCESSING CONFIGURATION
# ==========================================
EN_NEGATION_PATTERNS = [
    r'\b(no|not|never|don\'t|doesn\'t|didn\'t|without)\s+([a-z\s]+)',
]

ENGLISH_STOP_WORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
    'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
    'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
    'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and',
    'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
    'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above',
    'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'nor', 'only',
    'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just', 'should', 'now',
    'feeling', 'feel', 'feels', 'suffering', 'help', 'please', 'doctor'
}

# ==========================================
# 3. MORPHOLOGY ANALYSIS FUNCTION (ထပ်ထည့်ထားသော ကဏ္ဍ)
# ==========================================
def lemmatize_word(word: str) -> str:
    """
    Morphology အရ စာလုံးများကို မူရင်း Form (Base Form) သို့ ပြောင်းပေးသော Function
    ဥပမာ - 'illed' -> 'ill', 'ate' -> 'eat', 'fevers' -> 'fever'
    """
    # 1. Verb (ကိရိယာ) အနေဖြင့် စစ်ဆေး၍ မူရင်းပြောင်းခြင်း (ate -> eat, running -> run)
    base_verb = lemmatizer.lemmatize(word, pos=wordnet.VERB)
    if base_verb != word:
        return base_verb
    
    # 2. Adjective နှင့် Noun အနေဖြင့် စစ်ဆေးခြင်း (illed -> ill, fevers -> fever)
    base_noun = lemmatizer.lemmatize(word, pos=wordnet.NOUN)
    base_verb = lemmatizer.lemmatize(word, pos=wordnet.VERB)
    base_adj = lemmatizer.lemmatize(word, pos=wordnet.ADJ)
    base_adv = lemmatizer.lemmatize(word, pos=wordnet.ADV)    
    
    if base_adj != word:
        return base_adj
    return base_noun


def handle_negation(text: str, lang: str = "en") -> str:
    """
    မဖြစ်ပွားသော လက္ခဏာများကို စာသားထဲမှ ကြိုတင်ဖယ်ထုတ်ပေးသည့် Function
    """
    if not isinstance(text, str):
        return ""
    
    patterns = MY_NEGATION_PATTERNS if lang == "my" else EN_NEGATION_PATTERNS
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
    return text


def detect_language(text: str) -> str:
    """
    Input Text ထဲတွင် မြန်မာ Unicode စာလုံးများ (U+1000 to U+109F) ပါမပါ စစ်ဆေးပေးသည်
    """
    if re.search(r'[\u1000-\u109F]', text):
        return "my"
    return "en"


def clean_text(text: str, lang: str = None) -> str:
    if not isinstance(text, str):
        return ""
    
    # ရိုက်ထည့်လိုက်သည့် Text ၏ ဘာသာစကားကို Auto Detect လုပ်သည်
    detected_lang = lang if lang else detect_language(text)
    
    text = text.lower().strip()
    
    # Detect ရရှိသော ဘာသာစကား (en သို့မဟုတ် my) အလိုက် Negation Handling ပြုလုပ်သည်
    text = handle_negation(text, lang=detected_lang)
    
    # သင်္ကေတများကို ရှင်းလင်းပါ
    text = re.sub(r'[၊။!@#$%^&*()_+\-=\[\]{};:\'",.<>/?\\|`~]', ' ', text)
    
    # Detect ရရှိသော ဘာသာစကားအလိုက် Stop Words ဖယ်ထုတ်သည်
    words = text.split()
    stop_words = MYANMAR_STOP_WORDS if detected_lang == "my" else ENGLISH_STOP_WORDS
    filtered = [w for w in words if w not in stop_words]
    
    # Morphology (Lemmatization) ပြုလုပ်ခြင်း (အင်္ဂလိပ်စာအတွက်သာ)
    if detected_lang == "en":
        filtered = [lemmatize_word(w) for w in filtered]
    
    return " ".join(filtered)


def normalize_symptom_name(symptom: str) -> str:
    if not isinstance(symptom, str):
        return ""
    return symptom.strip().replace(" ", "_")


def predict_next_words(user_input: str, candidates: list, top_n: int = 5) -> list:
    """
    မြန်မာစာနှင့် အင်္ဂလိပ်စာ နှစ်မျိုးစလုံးအတွက် 
    User ရိုက်ထည့်လိုက်သော စာသားကို အခြေခံ၍ နောက်လာမည့် စာလုံး/ရောဂါအမည်ကို Predict လုပ်ပေးသည့် function
    """
    if not user_input or not isinstance(user_input, str):
        return []
    
    query = user_input.strip()
    query_lower = query.lower()
    predictions = []
    
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
            
        cand_clean = candidate.strip()
        cand_lower = cand_clean.lower()
        
        if cand_clean.startswith(query) or cand_lower.startswith(query_lower):
            predictions.append((cand_clean, 2))
        elif query in cand_clean or query_lower in cand_lower:
            predictions.append((cand_clean, 1))
            
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    result = []
    for item in predictions:
        if item[0] not in result:
            result.append(item[0])
            if len(result) == top_n:
                break
                
    return result