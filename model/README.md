---
license: apache-2.0
base_model: bert-base-cased

datasets:
- gretelai/symptom_to_diagnosis
metrics:
- f1
tags:
- medical
widget:
- text: >-
    I've been having a lot of pain in my neck and back. I've also been having
    trouble with my balance and coordination. I've been coughing a lot and my
    limbs feel weak.
- text: >-
    I've been feeling really run down and weak. My throat is sore and I've been
    coughing a lot. I've also been having chills and a fever.

model-index:
- name: Symptom_to_Diagnosis
  results:
  - task:
      type: text-classification

    dataset:
      type: gretelai/symptom_to_diagnosis
      name: gretelai/symptom_to_diagnosis
      split: test
    metrics:
    - type: precision
      value: 0.94
      name: macro avg
    
    - type: recall
      value: 0.93
      name: macro avg
    
    - type: f1-score
      value: 0.93
      name: macro avg
language:

- en

---

# Symptom_to_Diagnosis

This model is a fine-tuned version of [bert-base-cased](https://huggingface.co/bert-base-cased)
on this dataset (https://huggingface.co/datasets/gretelai/symptom_to_diagnosis).

## Model description

Model Description
This model is a fine-tuned version of the bert-base-cased architecture, 
specifically designed for text classification tasks related to diagnosing diseases from symptoms. 
The primary objective is to analyze natural language descriptions of symptoms and predict one of 22 corresponding diagnoses. 

## Dataset Information
The model was trained on the Gretel/symptom_to_diagnosis dataset, which consists of 1,065 symptom descriptions in the English language, 
each labeled with one of the 22 possible diagnoses. The dataset focuses on fine-grained single-domain diagnosis, 
making it suitable for tasks that require detailed classification based on symptom descriptions.
Example

{
  "output_text": "drug reaction",
  "input_text": "I've been having headaches and migraines, and I can't sleep. My whole body shakes and twitches. Sometimes I feel lightheaded."
}


# Use a pipeline as a high-level helper
```
from transformers import pipeline

pipe = pipeline("text-classification", model="Zabihin/Symptom_to_Diagnosis")

Example:
result = pipe("I've been having headaches and migraines, and I can't sleep. My whole body shakes and twitches. Sometimes I feel lightheaded.")
result:

[{'label': 'drug reaction', 'score': 0.9489321112632751}]
```

or 

```
from transformers import pipeline

# Load the model
classifier = pipeline("text-classification", model="Zabihin/Symptom_to_Diagnosis", tokenizer="Zabihin/Symptom_to_Diagnosis")

# Example input text
input_text = "I've been having headaches and migraines, and I can't sleep. My whole body shakes and twitches. Sometimes I feel lightheaded."

# Get the predicted label
result = classifier(input_text)

# Print the predicted label
predicted_label = result[0]['label']
print("Predicted Label:", predicted_label)

Predicted Label: drug reaction
```


### Framework versions

- Transformers 4.35.2
- TensorFlow 2.15.0
- Datasets 2.15.0
- Tokenizers 0.15.0
