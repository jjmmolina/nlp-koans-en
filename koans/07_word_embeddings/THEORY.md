# Theory: Word Embeddings

## What are Word Embeddings?

Dense vector representations of words that capture semantic meaning.

```
king - man + woman  queen
```

## Why Embeddings?

Traditional **one-hot encoding** doesn't capture meaning:
- Large, sparse vectors
- No semantic relationships

**Embeddings** solve this:
- Dense vectors (50-300 dimensions)
- Similar words have similar vectors
- Capture relationships and analogies

## Popular Methods

### Word2Vec (2013)

Two architectures:
- **CBOW**: Predicts word from context
- **Skip-gram**: Predicts context from word

```python
from gensim.models import Word2Vec

sentences = [["cat", "sits", "mat"], ["dog", "plays", "ball"]]
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1)

# Get vector
vector = model.wv["cat"]

# Find similar
similar = model.wv.most_similar("cat")
```

### GloVe (2014)

Global Vectors - uses word co-occurrence statistics.

### FastText (2016)

Considers subword information (good for rare words).

## Pre-trained Embeddings

Don't train from scratch - use pre-trained:
- **Word2Vec**: Google News (300d)
- **GloVe**: Wikipedia + Gigaword
- **FastText**: 157 languages

## Applications

- Document similarity
- Word analogies
- Feature extraction for ML
- Basis for modern transformers

**Practice with tests! **
