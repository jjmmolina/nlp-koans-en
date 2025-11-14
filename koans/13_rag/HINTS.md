# Hints: RAG (Retrieval-Augmented Generation)

## Common Issues

### Documents not being retrieved
- Check embedding model is working
- Verify documents were added to vectorstore
- Try increasing k (number of results)

### LLM not using retrieved context
- Improve prompt template
- Check context is being passed correctly
- Verify document format

### Poor answer quality
- Retrieve more documents (increase k)
- Use better embedding model
- Improve document chunking
- Try reranking

### Slow performance
- Cache embeddings
- Use smaller embedding model
- Reduce chunk size
- Limit number of retrieved docs

## Quick Solutions

**Basic LangChain RAG:**
```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# Load and split documents
loader = TextLoader("doc.txt")
documents = loader.load()
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = splitter.split_documents(documents)

# Create vectorstore
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(texts, embeddings)

# Create QA chain
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)

# Ask question
answer = qa.run("What is the main topic?")
```

**Custom prompt:**
```python
from langchain.prompts import PromptTemplate

template = """Use the following context to answer the question.
Context: {context}
Question: {question}
Answer:"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])
```

**You can do it! **
