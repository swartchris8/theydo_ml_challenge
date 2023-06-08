import pathlib

from transformers import AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, AutoTokenizer
import torch

current_dir = pathlib.Path(__file__).parent.parent.resolve()
summarisation_dir = current_dir / "finetuned_summarisation"
translation_dir = current_dir / "finetuned_de-en_translation"
sentiment_dir = current_dir / "finetuned-imdb-bert"

tokenizer = AutoTokenizer.from_pretrained(summarisation_dir)
sum_model = AutoModelForSeq2SeqLM.from_pretrained(summarisation_dir)
translation_model = AutoModelForSeq2SeqLM.from_pretrained(translation_dir)

sentiment_tokenizer = AutoTokenizer.from_pretrained("finetuned-imdb-bert")
sentiment_model = AutoModelForSequenceClassification.from_pretrained("finetuned-imdb-bert")



def summarise(text: str) -> str:
    input_ids = tokenizer(f"summarize: {text.strip()}", return_tensors="pt").input_ids
    outputs = sum_model.generate(input_ids)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


tl_test = "Das Haus ist wunderbar."

def translate_german_to_english(text: str, task_prefix = "") -> str:
    inputs = tokenizer([f"{task_prefix}{sentence}" for sentence in [text]], return_tensors="pt", padding=True)
    outputs = translation_model.generate(inputs["input_ids"])
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def sentiment(text: str, with_prob=False) -> str:
    inputs = sentiment_tokenizer(text[:508], return_tensors="pt")
    
    with torch.no_grad():
    
        logits = sentiment_model(**inputs).logits
    
    predicted_class_id = logits.argmax().item()
    predicted_label = sentiment_model.config.id2label[predicted_class_id]
    probs = torch.nn.functional.softmax(logits)
    
    if with_prob:
        return {"label": predicted_label, "probability": probs}
    else:
        return predicted_label

def infer_text(text: str):
    return summarise(text)



