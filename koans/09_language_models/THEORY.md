# Theory: Language Models

## What is a Language Model?

A model that predicts the probability of word sequences.

```
P("The cat sits on the") = ?
Next word likely: "mat", "floor", "chair"
```

## Types

### N-gram Models
Statistical models using word sequences.

```python
P(word_i | word_{i-1}, word_{i-2}, ..., word_{i-n+1})
```

### Neural Language Models
Use neural networks to predict next word.

### Transformer Language Models
Modern approach (GPT, BERT, etc.)

## Key Concepts

### Perplexity
Measures how well model predicts text (lower = better).

### Temperature
Controls randomness in generation:
- Low (0.1): Deterministic
- High (1.5): Creative

## Modern LLMs

### GPT (Generative Pre-trained Transformer)
- Decoder-only
- Autoregressive (predicts next token)
- Trained on massive text

```python
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")
text = generator("Once upon a time", max_length=50)
```

### BERT vs GPT

- **BERT**: Bidirectional, good for understanding
- **GPT**: Unidirectional, good for generation

## Applications

- **Text generation**: Stories, code, emails
- **Completion**: Autocomplete
- **Chatbots**: Conversational AI
- **Translation**: Machine translation

**Practice with tests! **
