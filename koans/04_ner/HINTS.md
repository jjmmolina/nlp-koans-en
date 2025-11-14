# Hints: Named Entity Recognition

## Common Issues

### spaCy model not found
```python
python -m spacy download en_core_web_sm
```

### Entities not detected
- Try larger models: en_core_web_md, en_core_web_lg
- Some entities need context
- Model might not know specific names

### NLTK NER requires multiple downloads
```python
nltk.download('maxent_ne_chunker')
nltk.download('words')
```

## Quick Solutions

**spaCy (recommended):**
```python
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
entities = [(ent.text, ent.label_) for ent in doc.ents]
```

**Filter by entity type:**
```python
persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
```

**NLTK:**
```python
from nltk import ne_chunk, pos_tag, word_tokenize
tree = ne_chunk(pos_tag(word_tokenize(text)))
```

**You can do it! **
