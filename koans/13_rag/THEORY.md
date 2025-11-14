# Theory: RAG (Retrieval-Augmented Generation)

## What is RAG?

Combining **retrieval** (search) with **generation** (LLM) to create accurate, grounded responses.

## The Problem RAG Solves

LLMs have limitations:
- **Hallucinations**: Make up facts
- **Outdated**: Training data cutoff
- **No access**: Can't see your private data

RAG fixes this by retrieving relevant facts first!

## How RAG Works

```
1. User asks question
2. Retrieve relevant documents from knowledge base
3. Pass documents + question to LLM
4. LLM generates answer grounded in documents
```

## Architecture

### 1. Indexing Phase
```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# Create embeddings
embeddings = OpenAIEmbeddings()

# Store documents
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings
)
```

### 2. Retrieval Phase
```python
# Find relevant docs
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
relevant_docs = retriever.get_relevant_documents("What is NLP?")
```

### 3. Generation Phase
```python
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    retriever=retriever
)

answer = qa_chain.run("What is NLP?")
```

## Advanced Techniques

### Chunking
Split documents into smaller pieces for better retrieval.

### Reranking
Re-score retrieved documents for relevance.

### Hybrid Search
Combine semantic search with keyword search.

### Citation
Include sources in answers.

## Frameworks

- **LangChain**: Most popular
- **LlamaIndex**: Specialized for RAG
- **Haystack**: Production-ready

## Applications

- **Customer support**: Answer from docs
- **Internal knowledge**: Company wikis
- **Legal/Medical**: Domain-specific QA
- **Research**: Literature review

**Practice with tests! **
