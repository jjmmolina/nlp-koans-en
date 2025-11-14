# Hints: Transformers

## Common Issues

### Model download slow/fails
Models can be large (500MB+). Use smaller models:
```python
pipeline("sentiment-analysis", model="distilbert-base-uncased")
```

### Out of memory
- Use smaller models
- Reduce batch size
- Use CPU instead of GPU for testing

### Token limit exceeded
Most models have max length (512 tokens). Truncate:
```python
tokenizer(text, truncation=True, max_length=512)
```

## Quick Solutions

**Basic pipeline:**
```python
from transformers import pipeline

classifier = pipeline("text-classification")
result = classifier("text")
```

**Custom model:**
```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
```

**Common tasks:**
- text-classification
- question-answering
- text-generation
- translation

**You can do it! **
