# Project Summary - NLP Koans

## Overview

**NLP Koans** is an interactive, test-driven learning project for Natural Language Processing (NLP) in Python. It combines the Koan teaching methodology with practical NLP exercises.

## Project Information

- **Name**: NLP Koans (English Version)
- **Repository**: [jjmmolina/nlp-koans-en](https://github.com/jjmmolina/nlp-koans-en)
- **License**: MIT
- **Language**: Python 3.8+
- **Original Version**: [nlp-koans (Spanish)](https://github.com/jjmmolina/nlp-koans)

## Key Features

###  Learning Methodology
- **Test-Driven Development (TDD)**: Learn by making tests pass
- **Progressive Difficulty**: From basic to advanced
- **Hands-On**: Implement real NLP solutions
- **Self-Paced**: Learn at your own speed

###  Content
- **13 Koans**: Covering fundamental to advanced NLP topics
- **4 Difficulty Levels**: Basic, Intermediate, Advanced, Expert
- **Multiple Libraries**: NLTK, spaCy, Transformers, LangChain
- **Modern Topics**: LLMs, Agents, RAG, Semantic Search

###  Technologies

#### Core Libraries
- **NLTK**: Classic NLP toolkit
- **spaCy**: Industrial-strength NLP
- **Transformers** (Hugging Face): Pre-trained models
- **scikit-learn**: Machine Learning
- **gensim**: Topic modeling and embeddings

#### Advanced Libraries (Koans 10-13)
- **OpenAI API**: GPT models
- **Anthropic API**: Claude
- **Ollama**: Local LLMs (free alternative)
- **LangChain**: Agent framework
- **LangGraph**: Agent orchestration
- **ChromaDB**: Vector database
- **FAISS**: Similarity search
- **Instructor**: Structured outputs
- **DSPy**: Programming over prompting

## Project Structure

```
nlp-koans-en/
 README.md                 # Project overview
 GUIDE.md                  # Complete guide
 FAQ.md                    # Frequently asked questions
 LEARNING_PATH.md          # Detailed learning path
 CONTRIBUTING.md           # Contribution guidelines
 PROJECT_SUMMARY.md        # This file
 requirements.txt          # Python dependencies
 pytest.ini                # pytest configuration

 koans/                    # All koans
     01_tokenization/      #  Basic
     02_stemming_lemmatization/
     03_pos_tagging/
     04_ner/
     05_text_classification/  #  Intermediate
     06_sentiment_analysis/
     07_word_embeddings/
     08_transformers/      #  Advanced
     09_language_models/
     10_modern_llms/       #  Expert
     11_ai_agents/
     12_semantic_search/
     13_rag/
```

## Koan Details

### Basic Level ( 6-8 hours)

#### Koan 01: Tokenization
- **Concepts**: Word/sentence tokenization, custom tokenizers
- **Libraries**: NLTK, spaCy
- **Skills**: Text preprocessing, multilingual support

#### Koan 02: Stemming & Lemmatization
- **Concepts**: Word normalization, morphological analysis
- **Libraries**: NLTK, spaCy
- **Skills**: Understanding word forms

#### Koan 03: POS Tagging
- **Concepts**: Part-of-speech tagging, linguistic features
- **Libraries**: spaCy, NLTK
- **Skills**: Grammar analysis, tag extraction

#### Koan 04: Named Entity Recognition
- **Concepts**: Entity extraction, NER systems
- **Libraries**: spaCy
- **Skills**: Information extraction

### Intermediate Level ( 8-10 hours)

#### Koan 05: Text Classification
- **Concepts**: Feature extraction, supervised learning
- **Libraries**: scikit-learn
- **Skills**: Building classifiers, evaluation

#### Koan 06: Sentiment Analysis
- **Concepts**: Transfer learning, transformers
- **Libraries**: transformers
- **Skills**: Using pre-trained models

#### Koan 07: Word Embeddings
- **Concepts**: Word vectors, semantic similarity
- **Libraries**: spaCy, gensim
- **Skills**: Vector operations, similarity metrics

### Advanced Level ( 8-10 hours)

#### Koan 08: Transformers
- **Concepts**: Attention mechanism, BERT
- **Libraries**: transformers
- **Skills**: Fine-tuning models

#### Koan 09: Language Models
- **Concepts**: Text generation, autoregressive models
- **Libraries**: transformers
- **Skills**: GPT models, controlled generation

### Expert Level ( 10-15 hours)

#### Koan 10: Modern LLMs & APIs
- **Concepts**: API integration, streaming, function calling
- **Libraries**: OpenAI, Anthropic, Ollama
- **Skills**: Working with production LLMs

#### Koan 11: AI Agents
- **Concepts**: ReAct pattern, tool use, memory
- **Libraries**: LangChain, LangGraph, DSPy
- **Skills**: Building autonomous agents

#### Koan 12: Semantic Search
- **Concepts**: Vector databases, similarity search
- **Libraries**: sentence-transformers, ChromaDB, FAISS
- **Skills**: Building search systems

#### Koan 13: RAG (Retrieval-Augmented Generation)
- **Concepts**: Document retrieval, generation
- **Libraries**: LangChain, ChromaDB, Instructor
- **Skills**: Building RAG pipelines, evaluation

## Target Audience

### Primary
- Python developers learning NLP
- Computer science students
- Data scientists entering NLP
- Software engineers adding NLP skills

### Prerequisites
-  Basic Python (functions, classes, lists)
-  Command line basics
-  Text editor or IDE

### Not Required
-  Previous NLP knowledge
-  Advanced mathematics
-  Deep Learning expertise

## Learning Outcomes

After completing all koans, students will be able to:

### Technical Skills
-  Preprocess text data
-  Build text classifiers
-  Use pre-trained models
-  Fine-tune transformers
-  Work with LLM APIs
-  Build AI agents
-  Implement semantic search
-  Create RAG applications

### Conceptual Understanding
-  NLP fundamentals
-  Machine Learning for text
-  Transformer architecture
-  Embeddings and vector search
-  Agent architectures
-  Production NLP systems

## Installation Requirements

### Minimum (Basic Koans)
- Python 3.8+
- 500 MB disk space
- 4 GB RAM

### Recommended (All Koans)
- Python 3.10+
- 5 GB disk space (models)
- 8 GB RAM
- GPU (optional, for faster training)

### Optional
- API keys for OpenAI, Anthropic (Koan 10)
- Ollama for local LLMs (free alternative)

## Testing

The project uses **pytest** for all testing:

```bash
# Run all tests
pytest

# Run specific koan
pytest koans/01_tokenization/ -v

# Run with coverage
pytest --cov=koans

# Run specific test
pytest koans/01_tokenization/test_tokenization.py::TestTokenizationBasics -v
```

## Documentation

### For Learners
- **README.md**: Quick start and overview
- **GUIDE.md**: Complete learning guide
- **FAQ.md**: Common questions
- **LEARNING_PATH.md**: Week-by-week plan
- **THEORY.md** (per koan): Concepts explained
- **HINTS.md** (per koan): Progressive hints

### For Contributors
- **CONTRIBUTING.md**: How to contribute
- **PROJECT_SUMMARY.md**: This file

## Maintenance

### Regular Updates
- Security patches
- Library updates
- New models
- Bug fixes

### Community
- GitHub Issues for bugs
- GitHub Discussions for questions
- Pull Requests welcome

## Future Plans

### Planned Features
- More koans (fine-tuning, deployment)
- Video tutorials
- Interactive online version
- More language translations

### Community Wishes
- VS Code extension
- Jupyter notebook version
- Mobile app

## Statistics

- **Total Koans**: 13
- **Total Files**: 88+
- **Lines of Code**: ~3,000+
- **Test Cases**: 100+
- **Estimated Completion Time**: 32-43 hours

## Comparison with Similar Projects

### vs Traditional Courses
-  Hands-on from day 1
-  Self-paced
-  Free and open source
-  Production-ready code

### vs Tutorials
-  Structured progression
-  Complete coverage
-  Built-in assessment (tests)
-  Active learning

### vs Books
-  Interactive
-  Modern tools and libraries
-  Regularly updated
-  Community support

## Success Stories

Students have used NLP Koans to:
- Land NLP engineer jobs
- Build production applications
- Complete university projects
- Start NLP careers

## Credits

### Original Author
- Jesus Martinez ([@jjmmolina](https://github.com/jjmmolina))

### Inspiration
- Ruby Koans
- Go Koans
- Zen Buddhism koans

### Contributors
- See [GitHub Contributors](https://github.com/jjmmolina/nlp-koans-en/graphs/contributors)

## License

MIT License - See [LICENSE](LICENSE) file for details.

Free to use for:
- Personal learning
- Teaching
- Commercial projects
- Distribution

## Contact

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and community
- **Email**: Available in GitHub profile

---

**Start your NLP journey today! **

Visit: [https://github.com/jjmmolina/nlp-koans-en](https://github.com/jjmmolina/nlp-koans-en)
