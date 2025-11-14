# Frequently Asked Questions (FAQ)

## Table of Contents

- [General Questions](#general-questions)
- [Installation and Setup](#installation-and-setup)
- [Using the Koans](#using-the-koans)
- [Technical Issues](#technical-issues)
- [Learning and Progress](#learning-and-progress)
- [Advanced Topics](#advanced-topics)

## General Questions

### What are NLP Koans?

NLP Koans is a learning project that combines:
- **Natural Language Processing (NLP)**: Text processing and analysis techniques
- **Test-Driven Development (TDD)**: Learn by making tests pass
- **Koan methodology**: Progressive, hands-on exercises

### Who is this project for?

- **Python developers** who want to learn NLP
- **Students** learning natural language processing
- **Professionals** looking to enter the NLP field
- **Anyone curious** about text processing

### Do I need previous NLP knowledge?

No! The project is designed to teach NLP from scratch. You only need basic Python knowledge.

### How long does it take to complete?

- **Basic level** (Koans 1-4): 6-8 hours
- **Intermediate** (Koans 5-7): 8-10 hours
- **Advanced** (Koans 8-9): 8-10 hours
- **Expert** (Koans 10-13): 10-15 hours
- **Total**: 32-43 hours (approximately 4-6 weeks at a comfortable pace)

### Is it free?

Yes! The project is completely free and open source (MIT License). Koans 10-13 also include free alternatives with Ollama (local LLMs) so you don'\''t need paid API keys.

## Installation and Setup

### What do I need to install?

**Minimum** (to start):
- Python 3.8+
- pytest
- nltk

**Complete** (all koans):
- All dependencies in `requirements.txt`
- spaCy models
- NLTK data

### Why doesn'\''t pip install work?

Common causes:
1. **Virtual environment not activated**
   ```bash
   # Activate first
   .\venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/Mac
   ```

2. **pip outdated**
   ```bash
   python -m pip install --upgrade pip
   ```

3. **Python version**
   ```bash
   python --version  # Must be 3.8 or higher
   ```

### Do I need to download models for each koan?

No. Download once and they'\''ll be available for all koans:
- **spaCy**: `python -m spacy download en_core_web_sm`
- **NLTK**: Run the download commands once

### Can I use Conda instead of venv?

Yes! You can use any virtual environment:
```bash
conda create -n nlp-koans python=3.10
conda activate nlp-koans
pip install -r requirements.txt
```

## Using the Koans

### In what order should I do the koans?

Follow the numerical order (01  13). Each koan builds on concepts from previous ones.

### Can I skip a koan?

Not recommended. Each koan teaches fundamental concepts used in subsequent ones.

### How do I know if I'\''ve completed a koan?

When all tests in that koan pass:
```bash
cd koans/01_tokenization
pytest -v
# All tests should show PASSED 
```

### What if I get stuck?

1. **Read the error carefully**: It tells you what'\''s wrong
2. **Check HINTS.md**: Progressive hints without spoiling the solution
3. **Read THEORY.md**: Review the concepts
4. **Experiment in Python REPL**: Test ideas interactively
5. **Check the tests**: They show what'\''s expected
6. **Last resort**: Look at Level 5 hints (complete solution)

### Can I modify the test files?

You can, but it'\''s not recommended. Tests define what you should learn. If tests seem wrong, open an issue on GitHub.

### Should I delete the TODO comments?

It'\''s optional. You can leave them or delete them after implementing the solution.

## Technical Issues

### Error: `ModuleNotFoundError`

**Problem**: Module not installed

**Solution**:
```bash
pip install <module_name>
# Or install everything:
pip install -r requirements.txt
```

### Error: `LookupError: Resource punkt not found`

**Problem**: NLTK data not downloaded

**Solution**:
```python
import nltk
nltk.download('\''punkt'\'')
# Or download all:
nltk.download('\''popular'\'')
```

### Error: `Can'\''t find model en_core_web_sm`

**Problem**: spaCy model not downloaded

**Solution**:
```bash
python -m spacy download en_core_web_sm
```

### Tests run but all fail

**Problem**: You haven'\''t implemented the functions yet

**Solution**: This is expected! Implement the functions in the .py files to make tests pass.

### Error: `AssertionError` in tests

**Problem**: Your implementation doesn'\''t match what'\''s expected

**Solution**: 
1. Read the error message - it tells you what'\''s wrong
2. Check the test to see what'\''s expected
3. Adjust your implementation

### pytest shows warnings

**Solution**: Warnings are normal. They don'\''t affect learning. You can ignore them or hide with:
```bash
pytest -v --disable-warnings
```

### Slow tests in advanced koans

**Solution**: Koans 8-13 use large models and may be slow. This is normal. You can:
- Use smaller models
- Run specific tests instead of all
- Be patient (first run downloads models)

## Learning and Progress

### How do I track my progress?

Run the progress checker:
```bash
# Windows
.\check_progress.ps1

# Linux/Mac
./check_progress.sh
```

### Can I do multiple koans per day?

Yes! Go at your own pace. Some people do one koan per day, others do multiple. The important thing is understanding, not speed.

### Should I take notes?

Yes! Taking notes helps consolidate learning. You can:
- Add comments to code
- Keep a learning journal
- Create your own examples

### How do I review what I'\''ve learned?

1. **Re-read THEORY.md** files
2. **Experiment** with the code you wrote
3. **Try new examples** with your own texts
4. **Explain** concepts to someone else
5. **Build** a small project using what you learned

### Is there a certificate?

No official certificate, but you'\''ll have:
- A complete project in your GitHub
- Practical NLP knowledge
- 13 working implementations
- Foundation to build real projects

## Advanced Topics

### Can I use these koans for teaching?

Yes! The project is MIT licensed. You can use it for:
- Personal learning
- Teaching courses
- Workshops
- University classes

Please give attribution and link to the original repository.

### Can I contribute?

Yes! Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

### Are there koans in other languages?

Currently available:
-  Spanish: [nlp-koans](https://github.com/jjmmolina/nlp-koans)
-  English: [nlp-koans-en](https://github.com/jjmmolina/nlp-koans-en)

### Do I need API keys for koans 10-13?

No! Koans 10-13 now include **Ollama** (local LLMs) so you can complete everything without API keys. Commercial APIs (OpenAI, Anthropic, Google) are optional for comparing different models.

### How do I use Ollama?

```bash
# Install Ollama (see ollama.ai)
# Then pull a model:
ollama pull llama2

# The koans will work with local models
```

### Can I use this to build real applications?

Yes! The concepts and code you learn are production-ready. After completing the koans, you'\''ll be able to:
- Build text classifiers
- Create chatbots
- Implement semantic search
- Build RAG applications
- Work with LLMs

### What'\''s next after completing all koans?

1. **Build a project**: Apply what you learned
2. **Explore advanced topics**: Fine-tuning, prompt engineering, etc.
3. **Contribute**: Add new koans or improve existing ones
4. **Learn more**: Take advanced courses, read research papers
5. **Join the community**: Participate in NLP forums and conferences

---

## Still have questions?

- **Open an issue**: [GitHub Issues](https://github.com/jjmmolina/nlp-koans-en/issues)
- **Start a discussion**: [GitHub Discussions](https://github.com/jjmmolina/nlp-koans-en/discussions)
- **Check the guide**: [GUIDE.md](GUIDE.md)

**Happy Learning! **
