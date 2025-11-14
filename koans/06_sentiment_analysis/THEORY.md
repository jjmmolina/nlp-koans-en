# Theory: Sentiment Analysis

## What is Sentiment Analysis?

Determining the emotional tone of text: **positive**, **negative**, or **neutral**.

## Approaches

### 1. Lexicon-Based

Uses predefined word lists with sentiment scores.

```python
from textblob import TextBlob

text = "I love this product!"
blob = TextBlob(text)
print(blob.sentiment.polarity)  # 0.5 (positive)
```

### 2. Machine Learning

Train models on labeled data.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

texts = ["I love it", "I hate it"]
labels = [1, 0]  # 1=positive, 0=negative

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

model = LogisticRegression()
model.fit(X, labels)
```

### 3. Deep Learning (Transformers)

Pre-trained models like BERT.

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I love this!")
# [{'label': 'POSITIVE', 'score': 0.99}]
```

## Challenges

- **Sarcasm**: "Great, another bug!" (negative despite "great")
- **Negation**: "not good" (must detect "not")
- **Context**: Same word, different meanings
- **Emoji**:  vs 

## Applications

- **Customer feedback**: Analyze reviews
- **Social media**: Monitor brand reputation
- **Market research**: Understand opinions
- **Customer service**: Prioritize complaints

**Practice with tests! **
