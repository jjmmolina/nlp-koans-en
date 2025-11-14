# Hints: Text Classification

## Common Issues

### Shape mismatch errors
Vectorizer must fit on training data, then transform test data:
```python
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)  # No fit!
```

### Poor accuracy
- Need more training data
- Try different features (TF-IDF vs CountVectorizer)
- Try different algorithms

### Memory errors
Reduce vocabulary size:
```python
CountVectorizer(max_features=1000)
```

## Quick Solutions

**Basic pipeline:**
```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

classifier = MultinomialNB()
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
```

**Evaluate:**
```python
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions)
```

**You can do it! **
