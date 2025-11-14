# Hints: Language Models

## Common Issues

### Generation loops or repeats
Use temperature and top_p:
```python
generator("text", max_length=50, temperature=0.7, top_p=0.9)
```

### Slow generation
- Use smaller models (distilgpt2)
- Reduce max_length
- Use num_return_sequences=1

### Nonsensical output
- Adjust temperature (lower = more deterministic)
- Try different prompts
- Check model is loaded correctly

## Quick Solutions

**Text generation:**
```python
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")
result = generator("Once upon a time", max_length=50)
```

**Control output:**
```python
result = generator(
    "text",
    max_length=100,
    num_return_sequences=3,
    temperature=0.8,
    top_k=50,
    top_p=0.95
)
```

**Perplexity calculation:**
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs, labels=inputs["input_ids"])
    perplexity = torch.exp(outputs.loss)
```

**You can do it! **
