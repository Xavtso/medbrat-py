import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import re
import json

# 1) TensorFlow-BERT classifier
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer as BFTokenizer
import tensorflow as tf

# 2) T5 + LoRA-adapter via PEFT for generation
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer as T5Tokenizer
from peft import PeftModel

class SymptomRequest(BaseModel):
    symptoms: str

app = FastAPI(
    title="Symptom→Diagnosis+Treatment (EN only)",
    version="1.0",
    description="Classify symptoms into 22 diagnoses and generate treatment recommendations in English."
)

# ─── 1) BERT classifier setup ────────────────────────────────────────
BF_MODEL_DIR = os.path.join(os.path.dirname(__file__), "model") 
id2label = {
    0: "allergy", 1: "arthritis", 2: "bronchial asthma", 3: "cervical spondylosis",
    4: "chicken pox", 5: "common cold", 6: "dengue", 7: "diabetes",
    8: "drug reaction", 9: "fungal infection", 10: "gastroesophageal reflux disease",
    11: "hypertension", 12: "impetigo", 13: "jaundice", 14: "malaria",
    15: "migraine", 16: "peptic ulcer disease", 17: "pneumonia",
    18: "psoriasis", 19: "typhoid", 20: "urinary tract infection", 21: "varicose veins"
}

bf_tokenizer = BFTokenizer.from_pretrained(BF_MODEL_DIR)
bf_model     = TFAutoModelForSequenceClassification.from_pretrained(BF_MODEL_DIR)


# ─── 2) T5 + LoRA via PEFT setup ─────────────────────────────────────
ADAPTER_DIR = os.path.join(os.path.dirname(__file__), "T5-medical_Symptom-Diagnoses")
BASE_T5     = "t5-small"

t5_tokenizer = T5Tokenizer.from_pretrained(BASE_T5)
base_t5      = AutoModelForSeq2SeqLM.from_pretrained(BASE_T5)
rec_model    = PeftModel.from_pretrained(base_t5, ADAPTER_DIR)


def classify_symptoms(text: str):
    """Returns (diagnosis_label, confidence) from BERT."""
    inputs = bf_tokenizer(text, return_tensors="tf", truncation=True, max_length=128)
    logits = bf_model(**inputs).logits
    idx    = int(tf.math.argmax(logits, axis=-1)[0])
    conf   = float(tf.nn.softmax(logits, axis=-1)[0, idx].numpy())
    return id2label[idx], conf





import re
import logging

logger = logging.getLogger(__name__)

DRUG_FALLBACK = {
    "allergy":      ["Cetirizine", "Loratadine", "Fexofenadine"],
    "arthritis":    ["Ibuprofen", "Naproxen", "Diclofenac"],
    "bronchial asthma": ["Albuterol", "Salmeterol", "Metformin"],
    "cervical spondylosis": ["Ibuprofen", "Naproxen", "Diclofenac"],
    "chicken pox":  ["Ibuprofen", "Naproxen", "Diclofenac"],
    "common cold":  ["Ibuprofen", "Naproxen", "Diclofenac"],
    "dengue":       ["Ibuprofen", "Naproxen", "Diclofenac"],
    "diabetes":     ["Metformin", "Glipizide", "Repaglinide"],
    "drug reaction": ["Ibuprofen", "Naproxen", "Diclofenac"],
    "fungal infection": ["Ibuprofen", "Naproxen", "Diclofenac"],
    "gastroesophageal reflux disease": ["Ibuprofen", "Naproxen", "Diclofenac"],
    "hypertension": ["Metformin", "Glipizide", "Repaglinide"],
    "impetigo":     ["Ibuprofen", "Naproxen", "Diclofenac"],
    "jaundice":     ["Ibuprofen", "Naproxen", "Diclofenac"],
    "malaria":      ["Ibuprofen", "Naproxen", "Diclofenac"],
    "migraine":     ["Ibuprofen", "Naproxen", "Diclofenac"],
    "peptic ulcer disease": ["Ibuprofen", "Naproxen", "Diclofenac"],
    "pneumonia":    ["Ibuprofen", "Naproxen", "Diclofenac"],
    "psoriasis":    ["Ibuprofen", "Naproxen", "Diclofenac"],
    "typhoid":      ["Ibuprofen", "Naproxen", "Diclofenac"],
    "urinary tract infection": ["Ibuprofen", "Naproxen", "Diclofenac"],
    "varicose veins": ["Ibuprofen", "Naproxen", "Diclofenac"],
}

def generate_drug_list(symptoms: str, diagnosis: str) -> str:
    prompt = (
        "You are a medical assistant.\n"
        f"Confirmed diagnosis: {diagnosis}.\n"
        "List exactly three generic drug names, separated by commas, with NO other text.\n"
        f"Patient symptoms: {symptoms}\n"
        "Drugs:"
    )
    tok = t5_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding="longest",
        max_length=256
    )
    outputs = rec_model.generate(
        input_ids=tok.input_ids,
        attention_mask=tok.attention_mask,
        max_length=tok.input_ids.shape[-1] + 20,
        num_beams=3,
        do_sample=False,
        early_stopping=True
    )

    raw = t5_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    logger.info(f"[generate_drug_list] raw: {raw!r}")

    # Витягуємо текст після останнього "Drugs:"
    match = re.search(r"[Dd]rugs\s*:\s*(.*)", raw)
    raw_list = match.group(1) if match else raw

    # Сплітимо за комами, очистимо крапки
    candidates = [re.sub(r"[.]*$", "", s.strip()) for s in raw_list.split(",")]

    # Видалимо небажані елементи та дублікати
    seen = set()
    drugs = []
    for item in candidates:
        if not item or item.lower().startswith("diagnosis"):
            continue
        if item not in seen:
            seen.add(item)
            drugs.append(item)
        if len(drugs) == 3:
            break

    logger.info(f"[generate_drug_list] parsed drugs: {drugs}")

    # Фолбек лише якщо парсинг НЕ дав жодної назви
    if len(drugs) == 0:
        logger.warning(f"[generate_drug_list] no drugs parsed, using fallback for {diagnosis}")
        drugs = DRUG_FALLBACK.get(diagnosis, [])[:3]

    # Повертаємо через кому (може бути 1–3 елементи)
    return ", ".join(drugs)




with open("recommendations_structured.json", encoding="utf-8") as f:
    STRUCTURED_RECS = json.load(f)

@app.post("/predict")
def assist(req: SymptomRequest):
    symptoms = req.symptoms.strip()
    if not symptoms:
        raise HTTPException(400, "Please provide symptoms in English.")

    # 1) classify
    diag_en, conf = classify_symptoms(symptoms)

    # 2) get drug list only
    drugs = generate_drug_list(symptoms, diag_en)

    # 2) recommendations — зі STRUCTURED_RECS
    rec_block = STRUCTURED_RECS.get(diag_en, {})

    return {
        "diagnosis":                   diag_en,
        "confidence":                  round(conf, 3),
        "drugs":                       drugs,
        "recommendations":             rec_block.get("recommendations", ""),
        "self_care":                   rec_block.get("self_care", ""),
        "reason_for_doctor_visit":     rec_block.get("reasonForDoctorVisit", "")
    }



if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
