# Hints: Stemming & Lemmatization

## Common Issues

### WordNetLemmatizer not working correctly
Need to specify POS tag:
```python
lemmatizer.lemmatize("running", pos="v")  # verb
lemmatizer.lemmatize("better", pos="a")   # adjective
```

### Missing WordNet data
```python
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
```

### Stemmer producing weird results
This is normal! Stemmers use heuristic rules, not dictionaries.

## Quick Solutions

**Stemming:**
```python
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
stem = stemmer.stem(word)
```

**Lemmatization:**
```python
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
lemma = lemmatizer.lemmatize(word, pos="v")
```

**spaCy lemmatization:**
```python
doc = nlp(text)
lemmas = [token.lemma_ for token in doc]
```

**You can do it! **
