# Theory: POS Tagging

## What is POS Tagging?

**Part-of-Speech (POS) Tagging** assigns grammatical tags to words.

```
"The cat sat on the mat"
[(The, DET), (cat, NOUN), (sat, VERB), (on, ADP), (the, DET), (mat, NOUN)]
```

## Common Tags

- **NOUN**: person, place, thing
- **VERB**: action or state
- **ADJ**: describes nouns
- **ADV**: modifies verbs/adjectives
- **DET**: the, a, this
- **PRON**: I, you, he, she

## Tools

### NLTK
```python
import nltk
tokens = nltk.word_tokenize("The cat sat")
tags = nltk.pos_tag(tokens)
# [('The', 'DT'), ('cat', 'NN'), ('sat', 'VBD')]
```

### spaCy
```python
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("The cat sat")
for token in doc:
    print(token.text, token.pos_)
```

## Why is it useful?

- Disambiguation: "record" (noun) vs "record" (verb)
- Feature extraction for ML
- Grammar checking
- Information extraction

**Practice with tests! **
