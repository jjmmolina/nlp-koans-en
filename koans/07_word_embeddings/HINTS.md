# Hints: Word Embeddings

## Common Issues

### Gensim installation
```bash
pip install gensim
```

### KeyError: word not in vocabulary
Check if word exists:
```python
if "word" in model.wv:
    vector = model.wv["word"]
```

### Training takes too long
Use smaller vector_size or fewer sentences:
```python
Word2Vec(sentences, vector_size=50, min_count=1)
```

### Loading pre-trained models
```python
from gensim.models import KeyedVectors
model = KeyedVectors.load_word2vec_format('path/to/model.bin', binary=True)
```

## Quick Solutions

**Train Word2Vec:**
```python
from gensim.models import Word2Vec
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1)
```

**Get vector:**
```python
vector = model.wv["word"]
```

**Find similar words:**
```python
similar = model.wv.most_similar("word", topn=5)
```

**Word similarity:**
```python
similarity = model.wv.similarity("cat", "dog")
```

**You can do it! **
