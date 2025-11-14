# Learning Path - NLP Koans

## Overview

This learning path is designed to guide you through the NLP Koans in an optimal way, maximizing learning and retention.

## Table of Contents

- [Time Commitment](#time-commitment)
- [Learning Stages](#learning-stages)
- [Detailed Path](#detailed-path)
- [Learning Strategies](#learning-strategies)
- [Assessment](#assessment)

## Time Commitment

### Total Estimated Time: 32-43 hours

| Level | Koans | Hours | Weeks (2h/day) |
|-------|-------|-------|----------------|
|  Basic | 1-4 | 6-8 | 1 |
|  Intermediate | 5-7 | 8-10 | 1-1.5 |
|  Advanced | 8-9 | 8-10 | 1-1.5 |
|  Expert | 10-13 | 10-15 | 1.5-2 |

**Recommended pace**: 2 hours per day, 5 days per week = 4-6 weeks total

## Learning Stages

### Stage 1: Foundations (Week 1)

**Objective**: Master basic NLP operations

**Koans**:
- Koan 01: Tokenization (2h)
- Koan 02: Stemming & Lemmatization (2h)
- Koan 03: POS Tagging (2h)
- Koan 04: Named Entity Recognition (2h)

**Skills acquired**:
- Text preprocessing
- Linguistic analysis
- Using NLTK and spaCy
- Understanding tokens and tags

### Stage 2: Applications (Week 2)

**Objective**: Apply ML to text

**Koans**:
- Koan 05: Text Classification (3h)
- Koan 06: Sentiment Analysis (2-3h)
- Koan 07: Word Embeddings (3-4h)

**Skills acquired**:
- Feature extraction
- Training classifiers
- Using transformers
- Word vector representations

### Stage 3: Deep Learning (Week 3)

**Objective**: Modern NLP with neural networks

**Koans**:
- Koan 08: Transformers (4-5h)
- Koan 09: Language Models (4-5h)

**Skills acquired**:
- Using Hugging Face
- Fine-tuning models
- Text generation
- Understanding attention mechanisms

### Stage 4: Production (Week 4)

**Objective**: Build real-world applications

**Koans**:
- Koan 10: Modern LLMs & APIs (2-3h)
- Koan 11: AI Agents (3-4h)
- Koan 12: Semantic Search (2-3h)
- Koan 13: RAG (3-5h)

**Skills acquired**:
- Working with LLM APIs
- Building agents
- Vector databases
- Retrieval-Augmented Generation

## Detailed Path

### Week 1: Foundations

#### Day 1: Koan 01 - Tokenization
- **Morning** (1h): Read THEORY.md, install dependencies
- **Afternoon** (1h): Implement functions, run tests

**Key concepts**:
- Word tokenization
- Sentence tokenization
- Custom tokenizers
- Multi-language support

#### Day 2: Koan 02 - Stemming & Lemmatization
- **Morning** (1h): Understand stemming vs lemmatization
- **Afternoon** (1h): Implement with NLTK and spaCy

**Key concepts**:
- Stemming algorithms
- Lemmatization with context
- Language-specific differences

#### Day 3: Koan 03 - POS Tagging
- **Morning** (1h): Learn part-of-speech tags
- **Afternoon** (1h): Implement POS tagging

**Key concepts**:
- POS tag sets
- Statistical tagging
- Rule-based vs ML-based

#### Day 4: Koan 04 - Named Entity Recognition
- **Morning** (1h): Understand NER concepts
- **Afternoon** (1h): Extract entities with spaCy

**Key concepts**:
- Entity types
- Entity recognition
- Custom NER models

#### Day 5: Review Week 1
- **Morning** (1h): Review koans 1-4
- **Afternoon** (1h): Build a mini project combining concepts

**Mini Project**: Text analyzer that tokenizes, tags, and extracts entities

### Week 2: Applications

#### Day 1-2: Koan 05 - Text Classification
- **Day 1** (2h): Feature extraction, build classifier
- **Day 2** (1h): Evaluate and improve

**Key concepts**:
- TF-IDF vectorization
- Training classifiers
- Evaluation metrics

#### Day 3: Koan 06 - Sentiment Analysis
- **Morning** (1h): Load pre-trained models
- **Afternoon** (1-2h): Analyze sentiments

**Key concepts**:
- Transfer learning
- Using transformers
- Sentiment scales

#### Day 4-5: Koan 07 - Word Embeddings
- **Day 4** (2h): Understand embeddings, Word2Vec
- **Day 5** (1-2h): Similarity and operations

**Key concepts**:
- Word vectors
- Semantic similarity
- Vector arithmetic

### Week 3: Deep Learning

#### Day 1-2: Koan 08 - Transformers
- **Day 1** (2h): Load and use BERT
- **Day 2** (2-3h): Fine-tuning and evaluation

**Key concepts**:
- Transformer architecture
- Pre-training and fine-tuning
- Tokenization for transformers

#### Day 3-4: Koan 09 - Language Models
- **Day 3** (2h): Text generation with GPT-2
- **Day 4** (2-3h): Controlled generation

**Key concepts**:
- Autoregressive models
- Sampling strategies
- Prompt engineering basics

#### Day 5: Review Week 2-3
- Build a project using transformers

### Week 4: Production

#### Day 1: Koan 10 - Modern LLMs
- **Session** (2-3h): Work with OpenAI, Ollama APIs

**Key concepts**:
- API integration
- Streaming responses
- Function calling
- Structured outputs

#### Day 2-3: Koan 11 - AI Agents
- **Day 2** (2h): Agent basics, tools
- **Day 3** (1-2h): ReAct pattern, memory

**Key concepts**:
- Agent architectures
- Tool use
- Memory systems
- LangChain/LangGraph

#### Day 4: Koan 12 - Semantic Search
- **Session** (2-3h): Vector databases, similarity search

**Key concepts**:
- Embeddings for search
- Vector databases (ChromaDB, FAISS)
- Hybrid search

#### Day 5-6: Koan 13 - RAG
- **Day 5** (2h): Document loading and chunking
- **Day 6** (1-3h): RAG pipeline, evaluation

**Key concepts**:
- Retrieval-Augmented Generation
- Chunking strategies
- RAG evaluation

#### Day 7: Final Review
- Review all koans
- Build a complete application

## Learning Strategies

### Active Learning
1. **Don'\''t just copy code**: Understand each line
2. **Experiment**: Try different parameters
3. **Break things**: See what happens when you change code
4. **Ask questions**: Write down what you don'\''t understand

### Spaced Repetition
1. **Review previous koans** regularly
2. **Come back** to earlier topics after learning new ones
3. **Connect concepts** between koans

### Project-Based Learning
After every 2-3 koans, build a small project:

- **After Koans 1-2**: Text normalizer
- **After Koans 3-4**: Entity extractor
- **After Koans 5-6**: Document classifier
- **After Koans 7-8**: Semantic similarity tool
- **After Koans 9-10**: Chatbot
- **After Koans 11-13**: RAG application

### Learning Journal
Keep notes on:
- Key concepts learned
- Challenges faced
- Questions that arose
- Ideas for applications

## Assessment

### Self-Assessment Checklist

After each koan, ask yourself:

- [ ] Do I understand the theory?
- [ ] Can I explain it to someone else?
- [ ] Did all tests pass?
- [ ] Can I modify the code?
- [ ] Can I apply this to new examples?

### Milestone Projects

**Beginner** (After Koan 4):
- Build a text analyzer that tokenizes, tags, and extracts entities

**Intermediate** (After Koan 7):
- Build a document classifier with sentiment analysis

**Advanced** (After Koan 9):
- Build a text generation application

**Expert** (After Koan 13):
- Build a complete RAG application with agents

## Tips for Success

### Before Starting
1. **Set clear goals**: Know why you'\''re learning NLP
2. **Schedule time**: Block regular study hours
3. **Prepare environment**: Set up workspace and tools

### During Learning
1. **Stay consistent**: Better 1h daily than 7h once/week
2. **Take breaks**: Pomodoro technique (25min work, 5min break)
3. **Ask for help**: Use GitHub Discussions
4. **Share progress**: Post on social media, blog about it

### After Completing
1. **Build a portfolio project**
2. **Contribute to open source**
3. **Continue learning**: Read papers, take advanced courses
4. **Join the community**: NLP meetups, conferences

## Additional Resources

### While Learning
- [THEORY.md files](koans/) - Theory for each koan
- [HINTS.md files](koans/) - Progressive hints
- [FAQ.md](FAQ.md) - Common questions

### After Completing
- [Papers with Code](https://paperswithcode.com/) - Latest research
- [Hugging Face Course](https://huggingface.co/course) - Advanced transformers
- [Fast.ai NLP Course](https://www.fast.ai/) - Practical deep learning

---

**Remember**: Learning is not a race. Take your time to understand each concept deeply. 

**Good luck on your NLP journey! **
