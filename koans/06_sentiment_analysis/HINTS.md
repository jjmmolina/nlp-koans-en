# Hints: Sentiment Analysis

## Common Issues

### TextBlob installation
```bash
pip install textblob
python -m textblob.download_corpora
```

### Transformers model too slow
Use distilled models:
```python
pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
```

### Polarity score interpretation
- Positive: > 0
- Neutral: = 0
- Negative: < 0

### Sarcasm not detected
Most models struggle with sarcasm - this is a known limitation!

## Quick Solutions

**TextBlob:**
```python
from textblob import TextBlob
blob = TextBlob(text)
polarity = blob.sentiment.polarity  # -1 to 1
```

**Transformers:**
```python
from transformers import pipeline
classifier = pipeline("sentiment-analysis")
result = classifier(text)[0]
label = result['label']  # POSITIVE/NEGATIVE
score = result['score']   # confidence
```

**scikit-learn:**
```python
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
model = LogisticRegression()
model.fit(X, labels)
```

**You can do it! **
