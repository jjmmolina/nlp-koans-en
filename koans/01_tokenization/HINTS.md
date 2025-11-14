# Hints: Tokenization

## Common Issues

### "AttributeError: module has no attribute"
Make sure you've downloaded required data:
```python
import nltk
nltk.download('punkt')
```

### Empty tokens
Check if text is actually a string, not None or empty.

### Different token counts
Different tokenizers use different rules. This is normal!

## Quick Solutions

**Word tokenization:**
```python
from nltk.tokenize import word_tokenize
tokens = word_tokenize(text)
```

**Sentence tokenization:**
```python
from nltk.tokenize import sent_tokenize
sentences = sent_tokenize(text)
```

**Subword tokenization:**
```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer.tokenize(text)
```

## Remember

- Always download required NLTK data first
- Test with simple examples before complex text
- Read the test assertions carefully

**You can do it! **
