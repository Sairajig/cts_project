import os 
import torch
os.environ["HUGGING_FACE_API"] = "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"

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6", clean_up_tokenization_spaces=True)
model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")

text = "Your text here for summarization."
inputs = tokenizer(text, return_tensors="pt")

summary_ids = model.generate(inputs["input_ids"])  # Corrected this line
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print("Summary:", summary)
