# Theory: Semantic Search

## What is Semantic Search?

Search by **meaning**, not just keywords.

Traditional search: "apple fruit"  finds exact words
Semantic search: "apple fruit"  also finds "red delicious", "granny smith"

## How It Works

### 1. Create Embeddings
Convert text to vectors that capture meaning.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

documents = [
    "The cat sits on the mat",
    "A dog plays in the park",
    "Felines enjoy resting on carpets"
]

embeddings = model.encode(documents)
```

### 2. Calculate Similarity
Use cosine similarity to find matches.

```python
from sklearn.metrics.pairwise import cosine_similarity

query = "cat on rug"
query_embedding = model.encode([query])

similarities = cosine_similarity(query_embedding, embeddings)
# Document 1 and 3 will have high similarity!
```

## Vector Databases

Store and search embeddings efficiently:

- **Pinecone**: Managed service
- **Weaviate**: Open source
- **ChromaDB**: Lightweight
- **FAISS**: Facebook's library

```python
import chromadb

client = chromadb.Client()
collection = client.create_collection("docs")

collection.add(
    documents=documents,
    ids=["1", "2", "3"]
)

results = collection.query(
    query_texts=["cat on rug"],
    n_results=2
)
```

## Advantages

- **Multilingual**: Works across languages
- **Synonyms**: Finds related terms
- **Context**: Understands meaning
- **Fuzzy**: Handles typos

## Applications

- **Q&A systems**: Find relevant answers
- **Recommendation**: Similar items
- **Document search**: Enterprise search
- **RAG**: Retrieval for LLMs

**Practice with tests! **
