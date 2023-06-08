import pathlib

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

current_dir = pathlib.Path(__file__).parent.parent.resolve()
summarisation_dir = current_dir / "finetuned_summarisation"

tokenizer = AutoTokenizer.from_pretrained(summarisation_dir)
sum_model = AutoModelForSeq2SeqLM.from_pretrained(summarisation_dir)


def summarise(text: str) -> str:
    input_ids = tokenizer(f"summarize: {text.strip()}", return_tensors="pt").input_ids
    outputs = sum_model.generate(input_ids)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


tl_test = "Das Haus ist wunderbar."

task_prefix = "translate German to English: "

def translate_german_to_english(text: str) -> str:
    inputs = tokenizer([f"{task_prefix}{sentence}" for sentence in [text]], return_tensors="pt", padding=True)
    outputs = sum_model.generate(inputs["input_ids"])
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def infer_text(text: str):
    return summarise(text)



