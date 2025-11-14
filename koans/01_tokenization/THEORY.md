# Theory: Tokenization

##  Table of Contents
1. [Introduction](#introduction)
2. [Types of Tokenization](#types)
3. [Tools](#tools)

##  Introduction

**Tokenization** splits text into tokens (words, sentences, or subwords).

```
Text: "Hello, world!"
Tokens: ["Hello", ",", "world", "!"]
```

##  Types

### Word Tokenization
```python
from nltk.tokenize import word_tokenize
tokens = word_tokenize("Hello, world!")
```

### Sentence Tokenization
```python
from nltk.tokenize import sent_tokenize
sentences = sent_tokenize("Hello! How are you?")
```

### Subword (Modern)
Used in BERT, GPT - handles rare words better.

##  Tools

- **NLTK**: Educational, easy to learn
- **spaCy**: Fast, production-ready
- **Transformers**: State-of-the-art

**Practice with tests! **
