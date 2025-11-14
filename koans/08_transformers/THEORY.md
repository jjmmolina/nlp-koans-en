# Theory: Transformers

## What are Transformers?

Revolutionary architecture introduced in "Attention is All You Need" (2017).

## Key Innovation: Attention Mechanism

Unlike RNNs that process sequentially, transformers process all words simultaneously using **attention**.

```
"The cat sat on the mat"

Attention lets model focus on relevant words:
- "sat" attends to "cat" (subject)
- "sat" attends to "mat" (object)
```

## Architecture

### Encoder
Processes input text (e.g., BERT)

### Decoder
Generates output text (e.g., GPT)

### Encoder-Decoder
Both (e.g., T5, BART)

## Famous Models

- **BERT** (2018): Bidirectional encoder, great for understanding
- **GPT** (2018-2023): Decoder, great for generation
- **T5** (2019): Text-to-text, versatile
- **RoBERTa** (2019): Optimized BERT

## Using Transformers

```python
from transformers import pipeline

# Text classification
classifier = pipeline("text-classification")
result = classifier("This is amazing!")

# Question answering
qa = pipeline("question-answering")
result = qa(question="What is NLP?", context="NLP is...")

# Translation
translator = pipeline("translation_en_to_es")
result = translator("Hello world")
```

## Why Transformers Won?

- **Parallelization**: Faster training
- **Long-range dependencies**: Better context understanding
- **Pre-training**: Transfer learning at scale
- **Flexibility**: Works for many tasks

**Practice with tests! **
