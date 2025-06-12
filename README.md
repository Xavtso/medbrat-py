# 📘 “РЕКОМЕНДАЦІЙНА ПЛАТФОРМА ДЛЯ

ДІАГНОСТУВАННЯ ЗАХВОРЮВАНЬ”

---

## 👤 Автор

* **ПІБ**: Хаврона Віталій Романович
* **Група**: ФеП-42
* **Керівник**: доц. Сінькевич Олег Олександрович
* **Дата виконання**: 11.06.2025

---

## 📌 Загальна інформація

* **Тип проєкту:** API / Веб-сервіс
* **Мова програмування:** Python
* **Фреймворки та бібліотеки:**

  * FastAPI
  * Uvicorn
  * Transformers
  * TensorFlow
  * PEFT
  * Pydantic

---

## 🧠 Функціонал

1. Прийом POST-запиту з полем `symptoms` (англійський рядок).
2. Класифікація симптомів у один із 22 діагнозів за допомогою попередньо навченого BERT.
3. Генерація списку з трьох препаратів через T5 з LoRA-адаптером.
4. Повернення JSON з полями:

   * `diagnosis`
   * `confidence`
   * `drugs`
   * `recommendations`
   * `self_care`
   * `reason_for_doctor_visit`
5. Запуск через Uvicorn із автоматичним перезавантаженням.

---

## 🧱 Структура проєкту

| Файл / Директорія                 | Призначення                                                                  |
| --------------------------------- | ---------------------------------------------------------------------------- |
| `main.py`                         | Налаштування FastAPI-додатку, імпорт моделей, реалізація `/predict` endpoint |
| `model/`                          | Папка з попередньо навченим BERT-моделлю для класифікації                    |
| `T5-medical_Symptom-Diagnoses/`   | LoRA-адаптер для T5-моделі, що генерує перелік препаратів                    |
| `prompts.txt`                     | Приклади вхідних промптів для BERT і T5 моделей                              |
| `recommendations_structured.json` | Структуровані поради, самодопомога та причини звернення до лікаря            |
| `requirements.txt`                | Перелік залежностей                                                          |
| `.gitignore`                      | Файли та папки, що ігноруються Git                                           |

---

## ▶️ Інструкція зі встановлення

1. **Клонувати репозиторій**

   ```bash
   git clone https://github.com/Xavtso/medbrat-py.git
   cd medbrat-py
   ```

2. **Встановити залежності**

   ```bash
   pip install -r requirements.txt
   ```

3. **Розмістити моделі**

   * Скопіювати папку `model/` (BERT) у корінь проєкту
   * Скопіювати папку `T5-medical_Symptom-Diagnoses/` (LoRA-адаптер) у корінь проєкту

---

## ▶️ Запуск сервісу

```bash
uvicorn main:app --reload
```

---

## 🔌 Приклад API

### POST `/predict`

**Запит**

```json
{
  "symptoms": "I have itchy, watery eyes and sneezing every time I go outside near flowers or trees."
}
```

**Відповідь**

```json
{
  "diagnosis": "allergy",
  "confidence": 0.925,
  "drugs": "Cetirizine, Loratadine, Fexofenadine",
  "recommendations": "Avoid allergens and ventilate the room regularly.",
  "self_care": "Take antihistamines if needed and rinse your nose with saline solution.",
  "reason_for_doctor_visit": "If you experience severe swelling, widespread rash, or difficulty breathing."
}
```

---

## 🖱️ Як користуватися

1. Надішліть POST-запит на `/predict` з тілом

   ```json
   { "symptoms": "<your symptoms>" }
   ```
2. Отримайте JSON з діагнозом, упевненістю, переліком препаратів та порадами.
3. Використайте результат для попередньої оцінки та подальших кроків.

---

## 🧪 Усунення проблем

| Проблема                                    | Рішення                                                                                  |
| ------------------------------------------- | ---------------------------------------------------------------------------------------- |
| Сервер не стартує через відсутність моделей | Перевірити наявність папок `model/` та `T5-medical_Symptom-Diagnoses/` у корені проєкту. |
| Нестача пам’яті при завантаженні моделей    | Використовувати CPU замість GPU або зменшити розмір моделей.                             |
| Порожній вхідний рядок у `symptoms`         | Додати валідацію на ненульову довжину рядка перед обробкою.                              |

