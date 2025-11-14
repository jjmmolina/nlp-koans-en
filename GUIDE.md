# Complete Guide - NLP Koans

## Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [How to Use the Koans](#how-to-use-the-koans)
5. [Koan Structure](#koan-structure)
6. [Recommended Learning Path](#recommended-learning-path)
7. [Tips and Best Practices](#tips-and-best-practices)
8. [Troubleshooting](#troubleshooting)
9. [Additional Resources](#additional-resources)

## Introduction

Welcome to **NLP Koans**! This project will guide you through learning Natural Language Processing (NLP) using the **Koan** methodology, where:

- Tests **fail initially**
- You **implement** the code to make them pass
- You **learn** NLP concepts progressively

This guide will help you make the most of the learning experience.

## Prerequisites

### Required Knowledge
- **Python**: Basic level (variables, functions, classes, lists, dictionaries)
- **Command line**: Basic commands (cd, ls/dir, running scripts)

### Optional but Helpful
- Basic understanding of unit testing
- Experience with virtual environments
- Text editor or IDE usage

### Not Required
-  Previous NLP knowledge
-  Advanced mathematics
-  Machine Learning or Deep Learning

## Installation

### Quick Installation (5 minutes)

For starting with the first koans:

```bash
# Clone the repository
git clone https://github.com/jjmmolina/nlp-koans-en.git
cd nlp-koans-en

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
.\venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install basic dependencies
pip install pytest nltk
```

### Complete Installation

For all koans including advanced ones:

```bash
# Install all dependencies
pip install -r requirements.txt

# Download spaCy English model
python -m spacy download en_core_web_sm

# Download NLTK data
python -c \"import nltk; nltk.download('\''punkt'\''); nltk.download('\''stopwords'\''); nltk.download('\''averaged_perceptron_tagger'\''); nltk.download('\''wordnet'\''); nltk.download('\''omw-1.4'\''); nltk.download('\''punkt_tab'\'')\"
```

## How to Use the Koans

### Basic Workflow

Each koan follows this pattern:

1. **Read the theory** (`THEORY.md`)
2. **Run the tests** (they will fail)
3. **Read the error** carefully
4. **Implement the solution**
5. **Run the tests again** until they pass
6. **Experiment** with the code

### Practical Example

Let'\''s solve your first koan:

```bash
# Navigate to first koan
cd koans/01_tokenization

# Run a specific test
pytest test_tokenization.py::TestTokenizationBasics::test_tokenize_words_nltk_english -v
```

You'\''ll see an error:
```
FAILED - AssertionError: List should not be empty
```

Open `tokenization.py` and find:
```python
def tokenize_words_nltk(text: str) -> List[str]:
    # TODO: Implement
    return []
```

Implement the solution:
```python
def tokenize_words_nltk(text: str) -> List[str]:
    from nltk.tokenize import word_tokenize
    return word_tokenize(text)
```

Run the test again:
```bash
pytest test_tokenization.py::TestTokenizationBasics::test_tokenize_words_nltk_english -v
#  PASSED!
```

### Using Hints

Each koan has a `HINTS.md` file with progressive hints:

- **Level 1**: Conceptual hint
- **Level 2**: Technical hint
- **Level 3**: Code structure hint
- **Level 4**: Almost complete solution
- **Level 5**: Complete solution

Use hints only when stuck!

## Koan Structure

Each koan directory contains:

```
koans/01_tokenization/
 THEORY.md          # Theoretical explanation
 HINTS.md           # Progressive hints
 tokenization.py    # Code to implement
 test_tokenization.py  # Tests
 __init__.py
```

### Difficulty Levels

| Level | Koans | Estimated Time | Topics |
|-------|-------|----------------|--------|
|  Basic | 1-4 | 6-8 hours | Tokenization, Lemmatization, POS, NER |
|  Intermediate | 5-7 | 8-10 hours | Classification, Sentiment, Embeddings |
|  Advanced | 8-9 | 8-10 hours | Transformers, Language Models |
|  Expert | 10-13 | 10-15 hours | Modern LLMs, Agents, RAG |

## Recommended Learning Path

### Week 1: Fundamentals (Koans 1-4)
- **Day 1-2**: Koan 01 - Tokenization
- **Day 3-4**: Koan 02 - Stemming & Lemmatization
- **Day 5-6**: Koan 03 - POS Tagging
- **Day 7**: Koan 04 - Named Entity Recognition

### Week 2: Intermediate (Koans 5-7)
- **Day 1-3**: Koan 05 - Text Classification
- **Day 4-5**: Koan 06 - Sentiment Analysis
- **Day 6-7**: Koan 07 - Word Embeddings

### Week 3: Advanced (Koans 8-9)
- **Day 1-4**: Koan 08 - Transformers
- **Day 5-7**: Koan 09 - Language Models

### Week 4: Expert (Koans 10-13)
- **Day 1-2**: Koan 10 - Modern LLMs
- **Day 3-4**: Koan 11 - AI Agents
- **Day 5-6**: Koan 12 - Semantic Search
- **Day 7**: Koan 13 - RAG

## Tips and Best Practices

### Learning Tips

1. **Don'\''t skip koans**: Each builds on previous concepts
2. **Read errors carefully**: They tell you exactly what'\''s wrong
3. **Use the REPL**: Test code snippets in Python interactive mode
4. **Experiment**: Try different inputs and edge cases
5. **Read the theory first**: Understanding concepts makes coding easier

### Coding Best Practices

1. **One test at a time**: Don'\''t try to solve everything at once
2. **Run tests frequently**: Get immediate feedback
3. **Write clean code**: Even in exercises, maintain good style
4. **Add print statements**: Debug by printing intermediate results
5. **Compare with hints**: But only after trying yourself

### Time Management

- **Focus sessions**: 25-45 minute blocks
- **Take breaks**: Rest between koans
- **Review**: Go back and review completed koans
- **Don'\''t rush**: Understanding > Speed

## Troubleshooting

### Common Issues

#### Import Errors

```python
ModuleNotFoundError: No module named '\''nltk'\''
```

**Solution**: Install the module
```bash
pip install nltk
```

#### NLTK Data Not Found

```python
LookupError: Resource punkt not found
```

**Solution**: Download NLTK data
```python
import nltk
nltk.download('\''punkt'\'')
```

#### spaCy Model Not Found

```
Can'\''t find model '\''en_core_web_sm'\''
```

**Solution**: Download the model
```bash
python -m spacy download en_core_web_sm
```

#### Tests Don'\''t Run

```bash
pytest: command not found
```

**Solution**: Install pytest and activate venv
```bash
pip install pytest
```

### Getting Help

1. **Check HINTS.md**: Progressive hints for each koan
2. **Read THEORY.md**: Understand the concepts
3. **Check FAQ.md**: Common questions answered
4. **GitHub Issues**: Open an issue for bugs
5. **Discussions**: Ask questions in GitHub Discussions

## Additional Resources

### Official Documentation
- [NLTK Documentation](https://www.nltk.org/)
- [spaCy Documentation](https://spacy.io/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [scikit-learn](https://scikit-learn.org/)

### Recommended Books
- *Natural Language Processing with Python* (NLTK Book)
- *Speech and Language Processing* by Jurafsky & Martin

### Online Courses
- Stanford CS224N: NLP with Deep Learning
- Fast.ai: Practical Deep Learning for Coders

---

**Happy Learning! **

For more information, see:
- [README.md](README.md) - Project overview
- [FAQ.md](FAQ.md) - Frequently asked questions
- [LEARNING_PATH.md](LEARNING_PATH.md) - Detailed learning path
- [CONTRIBUTING.md](CONTRIBUTING.md) - How to contribute
