# Theory: Text Classification

## What is Text Classification?

Assigning predefined categories to text documents.

## Common Applications

- **Spam Detection**: spam vs legitimate
- **Sentiment Analysis**: positive/negative/neutral
- **Topic Classification**: sports, politics, technology
- **Language Detection**: identify language

## Machine Learning Approach

### 1. Feature Extraction

Transform text to numerical vectors:

```python
from sklearn.feature_extraction.text import CountVectorizer

texts = ["I love this", "I hate this"]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
```

### 2. Train Classifier

```python
from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB()
y = [1, 0]  # 1=positive, 0=negative
classifier.fit(X, y)
```

### 3. Predict

```python
new_text = ["This is great"]
X_new = vectorizer.transform(new_text)
prediction = classifier.predict(X_new)
```

## Common Algorithms

- **Naive Bayes**: Fast, good baseline
- **Logistic Regression**: Interpretable
- **SVM**: Effective for text
- **Neural Networks**: State-of-the-art

## Feature Extraction Methods

- **Bag of Words**: Word frequency
- **TF-IDF**: Term importance
- **Word Embeddings**: Semantic vectors

**Practice with tests! **
