# Theory: Stemming & Lemmatization

## What are they?

**Stemming** and **Lemmatization** normalize words to their base form.

### Stemming
Removes suffixes using rules (fast but imprecise).

```python
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
print(stemmer.stem("running"))  # run
print(stemmer.stem("studies"))  # studi (not perfect!)
```

### Lemmatization
Uses vocabulary and morphology (slower but accurate).

```python
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("running", pos="v"))  # run
print(lemmatizer.lemmatize("better", pos="a"))  # good
```

## When to use each?

- **Stemming**: Speed matters, precision less important
- **Lemmatization**: Need real words, better for meaning

## Tools

- **NLTK**: Porter, Snowball, Lancaster stemmers
- **spaCy**: Built-in lemmatization

**Practice with tests! **
