from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from transformers import (
    TFAutoModelForSequenceClassification, 
    AutoTokenizer
)
import tensorflow as tf
import os

# 1) Імпортуємо googletrans
from googletrans import Translator

class SymptomRequest(BaseModel):
    symptoms: str

app = FastAPI()

model = None
tokenizer = None
translator = Translator()

# Користувацький словничок для відображення деяких хвороб українською
EN_TO_UA = {
    "allergy": "Алергія",
    "arthritis": "Артрит",
    "bronchial asthma": "Бронхіальна астма",
    "cervical spondylosis": "Шийний спондильоз",
    "chicken pox": "Вітряна віспа",
    "common cold": "Застуда",
    "dengue": "Денге",
    "diabetes": "Діабет",
    "drug reaction": "Реакція на ліки",
    "fungal infection": "Грибкова інфекція",
    "gastroesophageal reflux disease": "Рефлюксна хвороба стравоходу",
    "hypertension": "Гіпертонія",
    "impetigo": "Імпетиго",
    "jaundice": "Жовтяниця",
    "malaria": "Малярія",
    "migraine": "Мігрень",
    "peptic ulcer disease": "Виразка шлунку/дванадцятипалої кишки",
    "pneumonia": "Пневмонія",
    "psoriasis": "Псоріаз",
    "typhoid": "Тиф",
    "urinary tract infection": "Інфекція сечовивідних шляхів",
    "varicose veins": "Варикозне розширення вен"
}

@app.on_event("startup")
def load_model():
    global model, tokenizer
    model_path = os.path.join(os.path.dirname(__file__), "model")
    print("Завантажую модель і токенайзер BERT...")

    # Завантажуємо токенайзер
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Завантажуємо TF-модель (tf_model.h5) для sequence classification
    # Без параметра from_tf, якщо це справді TensorFlow-модель
    model = TFAutoModelForSequenceClassification.from_pretrained(model_path)
    print("Модель та токенайзер успішно завантажені!")

@app.post("/predict")
def predict(data: SymptomRequest):
    """
    Приймає текст українською -> перекладає його англійською -> класифікує -> повертає діагноз українською.
    """
    uk_text = data.symptoms

    # 1) Переклад українського тексту на англійську
    translated_text = translator.translate(uk_text, src='uk', dest='en').text
    print(translated_text)

    # 2) Класифікація (BERT)
    inputs = tokenizer(
        translated_text,
        return_tensors="tf",
        truncation=True,
        max_length=128
    )

    outputs = model(**inputs)
    logits = outputs.logits  # Tensor, [batch_size, num_labels]

    predicted_class_id = int(tf.math.argmax(logits, axis=-1)[0])

    # Якщо модель має в config.json id2label
    # transformers можуть автоматично дістати labels,
    # але тут покажемо, як взяти вручну:
    id2label = {
        0: "allergy",
        1: "arthritis",
        2: "bronchial asthma",
        3: "cervical spondylosis",
        4: "chicken pox",
        5: "common cold",
        6: "dengue",
        7: "diabetes",
        8: "drug reaction",
        9: "fungal infection",
        10: "gastroesophageal reflux disease",
        11: "hypertension",
        12: "impetigo",
        13: "jaundice",
        14: "malaria",
        15: "migraine",
        16: "peptic ulcer disease",
        17: "pneumonia",
        18: "psoriasis",
        19: "typhoid",
        20: "urinary tract infection",
        21: "varicose veins"
    }

    en_label = id2label.get(predicted_class_id, "Unknown")
    
    # 3) Перекладемо назву хвороби на українську через словничок
    #    Якщо знайдено в EN_TO_UA — візьмемо звідти, інакше лишимо "Unknown"
    ua_label = EN_TO_UA.get(en_label, "Невідомо")

    return {
        "translated_input": translated_text,  # Щоб побачити, як переклав googletrans
        "en_diagnosis": en_label,
        "ua_diagnosis": ua_label
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
