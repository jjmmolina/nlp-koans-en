#  NLP Koans - Learn Natural Language Processing with TDD

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![pytest](https://img.shields.io/badge/tested%20with-pytest-orange.svg)](https://pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![spaCy](https://img.shields.io/badge/spaCy-3.7%2B-09a3d5.svg)](https://spacy.io/)
[![Transformers](https://img.shields.io/badge/-Transformers-yellow.svg)](https://huggingface.co/transformers/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

A **Koan-style** tutorial project for learning **Natural Language Processing (NLP)** using **Test-Driven Development (TDD)** in Python.

>  **Language versions**: [Español](https://github.com/jjmmolina/nlp-koans) | **English** (you are here)

##  What are NLP Koans?

**Koans** are learning exercises where:
1.  Tests **fail initially** 
2.  You **fix the code** to make them pass
3.  You **learn** NLP concepts progressively

##  Quick Start

###  Quick Start (5 minutes)

```bash
# 1. Clone the repository
git clone https://github.com/jjmmolina/nlp-koans-en.git
cd nlp-koans-en

# 2. Create virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Install basic dependencies
pip install pytest nltk

# 4. Start with the first koan!
pytest koans/01_tokenization/test_tokenization.py -v
# You will see failing tests - that is expected! 
```

###  Full Installation

To use ALL koans (including advanced ones):

```bash
# Install all dependencies (may take some time)
pip install -r requirements.txt

# Download spaCy models
python -m spacy download en_core_web_sm
python -m spacy download es_core_news_sm  # Optional: Spanish model

# Download NLTK resources
python -c "import nltk; nltk.download(''punkt''); nltk.download(''stopwords''); nltk.download(''averaged_perceptron_tagger''); nltk.download(''wordnet''); nltk.download(''omw-1.4''); nltk.download(''punkt_tab'')"

# Run all tests
pytest
```

>  **Tip**: Start with Quick Start. Install the rest when you reach advanced koans.

##  Koan Structure

###  Basic Level (Koans 1-4)
| Koan | Topic | Libraries | Concepts |
|------|-------|-----------|----------|
| **01** | Tokenization | NLTK, spaCy | Splitting text into words/sentences |
| **02** | Stemming & Lemmatization | NLTK, spaCy | Word normalization |
| **03** | POS Tagging | spaCy, NLTK | Part-of-speech tagging |
| **04** | Named Entity Recognition | spaCy | Entity extraction |

###  Intermediate Level (Koans 5-7)
| Koan | Topic | Libraries | Concepts |
|------|-------|-----------|----------|
| **05** | Text Classification | scikit-learn | Text classification |
| **06** | Sentiment Analysis | transformers | Sentiment analysis |
| **07** | Word Embeddings | spaCy, gensim | Vector representations |

###  Advanced Level (Koans 8-9)
| Koan | Topic | Libraries | Concepts |
|------|-------|-----------|----------|
| **08** | Transformers | transformers (Hugging Face) | Pre-trained models |
| **09** | Language Models | transformers | Text generation |

###  Expert Level - Modern LLMs (Koans 10-13)
| Koan | Topic | Libraries | Concepts |
|------|-------|-----------|----------|
| **10** | Modern LLMs & APIs | OpenAI, Anthropic, **Ollama** | GPT-4, Claude, Gemini, local LLMs, streaming, function calling, **structured outputs** |
| **11** | AI Agents | LangChain, LangGraph | ReAct pattern, tools, memory, callbacks, **DSPy** |
| **12** | Semantic Search | sentence-transformers, ChromaDB, FAISS | Embeddings, vector databases, semantic search, **hybrid search** |
| **13** | RAG | LangChain, ChromaDB, **Instructor** | Retrieval-Augmented Generation, chunking, **evaluation**, **observability** |

>  **2025 Updates**: Ollama for local LLMs (no API keys needed), Instructor for structured outputs, DSPy for automatic optimization, Guardrails AI for safety, LangSmith for observability.

##  How to Use This Tutorial

###  Your First Koan in 3 Steps

**Step 1: Run the test (you will see it fail)**
```bash
cd koans/01_tokenization
pytest test_tokenization.py::TestTokenizationBasics::test_tokenize_words_nltk_english -v
```

You will see:
```
FAILED - AssertionError: List should not be empty
```

**Step 2: Open `tokenization.py` and find:**
```python
def tokenize_words_nltk(text: str) -> List[str]:
    # TODO: Implement word tokenization with nltk.word_tokenize()
    # Hint: from nltk.tokenize import word_tokenize
    return []  #  This is wrong, returns empty list
```

**Step 3: Implement the solution:**
```python
def tokenize_words_nltk(text: str) -> List[str]:
    from nltk.tokenize import word_tokenize
    return word_tokenize(text)  #  Now it is correct!
```

**Verify:**
```bash
pytest test_tokenization.py::TestTokenizationBasics::test_tokenize_words_nltk_english -v
#  PASSED - Congratulations!
```

**Repeat this process with all functions!** 

##  Technologies and Libraries

- ** Python 3.8+**: Base language
- ** pytest**: Testing framework
- ** spaCy**: Industrial-strength NLP
- ** NLTK**: Classic Natural Language Toolkit
- ** transformers**: Hugging Face models
- ** scikit-learn**: Traditional Machine Learning
- ** gensim**: Topic modeling and embeddings

##  Additional Documentation

-  [**GUIDE.md**](GUIDE.md) - Detailed step-by-step guide
-  [**LEARNING_PATH.md**](LEARNING_PATH.md) - Optimized learning path with estimated times
-  [**FAQ.md**](FAQ.md) - Frequently asked questions and troubleshooting
-  [**CONTRIBUTING.md**](CONTRIBUTING.md) - How to contribute to the project
-  [**LICENSE**](LICENSE) - MIT License
-  [**PROJECT_SUMMARY.md**](PROJECT_SUMMARY.md) - Technical project summary

##  Recommended Order

It is recommended to follow the koan order (01  13) as each builds on previous concepts.

**Learning Levels**:
-  **Basic (Koans 1-4)**: NLP fundamentals - 6-8 hours
-  **Intermediate (Koans 5-7)**: ML applied to NLP - 8-10 hours  
-  **Advanced (Koans 8-9)**: Transformers and LLMs - 8-10 hours
-  **Expert (Koans 10-13)**: Modern APIs, Agents, RAG - 10-15 hours

>  **Koans 10-13 now include local alternatives with Ollama** (no API keys needed). Commercial API keys (OpenAI, Anthropic) are optional for comparing models.

>  **Tech Radar 2025**: The course incorporates techniques from Thoughtworks Technology Radar Vol. 33: DSPy (programming over prompting), Instructor (structured outputs), Guardrails AI (safety), LangSmith (observability), and Mem0 (personalized memory).

**Prerequisites**:
-  Basic Python (variables, functions, classes)
-  Basic understanding of testing (optional but helpful)

**You do not need to know**:
-  Previous NLP knowledge
-  Advanced mathematics
-  Deep Learning

##  Tips

1. **Do not skip koans**: Each teaches fundamental concepts
2. **Read the documentation**: Each koan has explanatory comments
3. **Experiment**: Try with your own texts
4. **Use VS Code**: Configured with tasks and debugging

##  VS Code Integration

This project is optimized for VS Code with:
-  Automatic testing configuration
-  Integrated debugging
-  Tasks to run individual koans

##  Quick Wins - Your First 30 Minutes

Want to see immediate results? Follow this:

### 1 Quick Setup (5 min)
```bash
git clone https://github.com/jjmmolina/nlp-koans-en.git
cd nlp-koans-en
python -m venv venv
.\venv\Scripts\activate  # Windows
pip install pytest nltk
```

### 2 Your First Victory (10 min)
```bash
cd koans/01_tokenization
pytest test_tokenization.py::TestCustomTokenization::test_custom_tokenize_spaces -v
```

Open `tokenization.py` and change:
```python
def custom_tokenize(text: str, delimiter: str = " ") -> List[str]:
    return []  #  WRONG
```

To:
```python
def custom_tokenize(text: str, delimiter: str = " ") -> List[str]:
    return text.split(delimiter)  #  CORRECT
```

Run the test again:
```bash
pytest test_tokenization.py::TestCustomTokenization::test_custom_tokenize_spaces -v
#  PASSED!
```

** Congratulations! You completed your first koan.**

### 3 Next Level (15 min)

Now implement `tokenize_words_nltk()`:
1. Read the `HINTS.md` file
2. Follow the hints level by level
3. Make the test pass

```bash
pytest test_tokenization.py::TestTokenizationBasics::test_tokenize_words_nltk_english -v
```

** You now master basic tokenization!**

---

**Continue with the rest of Koan 01 and you will be officially on your way to NLP mastery.** 

##  Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

##  License

MIT License - see [LICENSE](LICENSE) for details.

##  Inspiration

Project inspired by:
- Ruby Koans
- Go Koans
- The power of deliberate practice learning

---

**Enjoy learning NLP! **
