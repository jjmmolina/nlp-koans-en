# Contributing to NLP Koans

First off, thank you for considering contributing to NLP Koans! It'\''s people like you that make this project a great learning resource for everyone.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Contribution Guidelines](#contribution-guidelines)
- [Style Guide](#style-guide)
- [Commit Guidelines](#commit-guidelines)

## Code of Conduct

This project and everyone participating in it is governed by our commitment to:

- Be respectful and inclusive
- Welcome newcomers
- Accept constructive criticism gracefully
- Focus on what is best for the community

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues. When creating a bug report, include:

- **Clear title and description**
- **Steps to reproduce**
- **Expected vs actual behavior**
- **Environment details** (OS, Python version, etc.)
- **Code samples** if applicable

**Example**:
```markdown
**Bug**: Test fails on Windows with Python 3.11

**Steps to reproduce**:
1. Install dependencies on Windows
2. Run `pytest koans/01_tokenization`
3. See error

**Expected**: All tests pass
**Actual**: ImportError

**Environment**:
- OS: Windows 11
- Python: 3.11.5
- pytest: 7.4.0
```

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When suggesting:

- **Use a clear title**
- **Provide detailed description**
- **Explain why it would be useful**
- **Provide examples** if applicable

### Contributing Code

#### Types of Contributions

1. **Fix bugs** in existing koans
2. **Improve documentation**
3. **Add tests**
4. **Improve hints**
5. **Translate to other languages**
6. **Create new koans** (discuss first)

#### Pull Request Process

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Make your changes**
4. **Add tests** if applicable
5. **Update documentation**
6. **Commit your changes** (see commit guidelines)
7. **Push to your fork** (`git push origin feature/amazing-feature`)
8. **Open a Pull Request**

## Development Setup

### 1. Fork and Clone

```bash
# Fork on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/nlp-koans-en.git
cd nlp-koans-en

# Add upstream remote
git remote add upstream https://github.com/jjmmolina/nlp-koans-en.git
```

### 2. Set Up Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black mypy
```

### 3. Run Tests

```bash
# Run all tests
pytest

# Run specific koan tests
pytest koans/01_tokenization/ -v

# Run with coverage
pytest --cov=koans
```

## Contribution Guidelines

### Adding a New Koan

Before adding a new koan:

1. **Open an issue** to discuss the topic
2. **Check if it fits** the learning progression
3. **Get approval** from maintainers

Structure for a new koan:
```
koans/XX_topic_name/
 __init__.py
 THEORY.md         # Detailed theory
 HINTS.md          # Progressive hints (5 levels)
 topic_name.py     # Code to implement
 test_topic_name.py  # Comprehensive tests
```

### Improving Existing Koans

When improving koans:

- **Keep backward compatibility** with tests
- **Update all related files** (THEORY.md, HINTS.md)
- **Test thoroughly**
- **Update documentation**

### Writing Tests

Tests should be:

- **Clear**: Test one thing at a time
- **Descriptive**: Use meaningful test names
- **Comprehensive**: Cover edge cases
- **Independent**: Don'\''t depend on other tests

**Example**:
```python
def test_tokenize_words_handles_empty_string(self):
    """Test tokenization with empty string."""
    result = tokenize_words_nltk("")
    assert result == []

def test_tokenize_words_with_punctuation(self):
    """Test tokenization preserves punctuation."""
    result = tokenize_words_nltk("Hello, world!")
    assert "," in result
    assert "!" in result
```

### Documentation

Good documentation:

- **Uses clear language**
- **Includes examples**
- **Explains why, not just how**
- **Has proper formatting**

## Style Guide

### Python Code

Follow PEP 8 and these guidelines:

```python
# Good: Clear function names, docstrings, type hints
def tokenize_words_nltk(text: str) -> List[str]:
    """
    Tokenize text into words using NLTK.
    
    Args:
        text: Input text to tokenize
        
    Returns:
        List of word tokens
        
    Example:
        >>> tokenize_words_nltk("Hello world")
        ['\''Hello'\'', '\''world'\'']
    """
    from nltk.tokenize import word_tokenize
    return word_tokenize(text)
```

### Markdown

- Use **clear headings**
- Include **code examples**
- Use **lists** for readability
- Add **links** where relevant

### Comments

```python
# Good: Explains why
# Using NLTK tokenizer because it handles contractions better

# Bad: States the obvious
# Tokenizing the text
```

## Commit Guidelines

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

**Examples**:

```bash
# Good
feat(koan01): Add multilingual tokenization support

Add support for Spanish and French tokenization
in the first koan. Includes tests and documentation.

Closes #123

# Good
fix(koan05): Fix TF-IDF vectorization for empty documents

Handle edge case where document list is empty

# Good
docs(guide): Improve installation instructions

Add troubleshooting section for Windows users
```

## Getting Help

- **Questions?** Open a [Discussion](https://github.com/jjmmolina/nlp-koans-en/discussions)
- **Stuck?** Check [FAQ.md](FAQ.md)
- **Found a bug?** Open an [Issue](https://github.com/jjmmolina/nlp-koans-en/issues)

## Recognition

Contributors will be:

- Listed in the README
- Credited in commit history
- Appreciated by the community! 

## Thank You!

Your contributions make this project better for everyone learning NLP. Every contribution, no matter how small, is valuable.

**Happy Contributing! **
