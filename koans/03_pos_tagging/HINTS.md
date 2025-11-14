# Hints: POS Tagging

## Common Issues

### Need averaged_perceptron_tagger
```python
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
```

### Different tag sets
NLTK uses Penn Treebank tags (DT, NN, VBD)
spaCy uses Universal tags (DET, NOUN, VERB)

### Wrong POS for ambiguous words
Context matters! "record" can be noun or verb.

## Quick Solutions

**NLTK:**
```python
import nltk
tokens = nltk.word_tokenize(text)
tags = nltk.pos_tag(tokens)
```

**spaCy:**
```python
doc = nlp(text)
tags = [(token.text, token.pos_) for token in doc]
```

**Get just POS tags:**
```python
pos_tags = [tag for word, tag in tags]
```

**You can do it! **
