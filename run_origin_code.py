from transformers import DebertaV2Tokenizer, DebertaV2ForMaskedLM
import torch


tokenizer = DebertaV2Tokenizer.from_pretrained('lib\\deberta-v2-xlarge')
model = DebertaV2ForMaskedLM.from_pretrained('lib\\deberta-v2-xlarge')

inputs = tokenizer("The capital of France is [MASK].", return_tensors="pt")
labels = tokenizer("The capital of France is Paris.", return_tensors="pt")["input_ids"]

outputs = model(**inputs, labels=labels)
loss = outputs.loss
logits = outputs.logits
pred = tokenizer.decode(torch.argmax(logits, dim=-1)[0, -2])
print(pred, ' ', loss)
