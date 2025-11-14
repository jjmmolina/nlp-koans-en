# Theory: Named Entity Recognition (NER)

## What is NER?

**NER** identifies and classifies named entities in text.

```
"Apple was founded by Steve Jobs in California"
[Apple: ORG, Steve Jobs: PERSON, California: GPE]
```

## Common Entity Types

- **PERSON**: People names
- **ORG**: Organizations, companies
- **GPE**: Countries, cities (Geo-Political Entity)
- **DATE**: Dates and periods
- **MONEY**: Monetary values
- **LOC**: Non-GPE locations

## Tools

### spaCy (Recommended)
```python
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple was founded by Steve Jobs")

for ent in doc.ents:
    print(ent.text, ent.label_)
# Apple ORG
# Steve Jobs PERSON
```

### NLTK
```python
import nltk
from nltk import ne_chunk, pos_tag, word_tokenize

text = "Apple was founded by Steve Jobs"
tokens = word_tokenize(text)
tags = pos_tag(tokens)
entities = ne_chunk(tags)
```

## Applications

- **Information Extraction**: Extract people, companies
- **Question Answering**: Find relevant entities
- **Content Classification**: Categorize by entities
- **Knowledge Graphs**: Build relationships

**Practice with tests! **
