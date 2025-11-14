# Hints: Semantic Search

## Common Issues

### sentence-transformers installation
```bash
pip install sentence-transformers
```

### Slow embedding creation
- Use smaller models
- Batch encode multiple texts
- Cache embeddings

### Poor search results
- Try different embedding models
- Adjust number of results (k parameter)
- Use hybrid search (keywords + semantic)

### Memory issues with large datasets
- Use vector databases (ChromaDB, Pinecone)
- Process in batches
- Use dimensionality reduction

## Quick Solutions

**Basic semantic search:**
```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings
docs = ["text1", "text2", "text3"]
doc_embeddings = model.encode(docs)

# Search
query = "search query"
query_embedding = model.encode([query])
similarities = cosine_similarity(query_embedding, doc_embeddings)[0]

# Get top results
top_indices = similarities.argsort()[-3:][::-1]
```

**With ChromaDB:**
```python
import chromadb

client = chromadb.Client()
collection = client.create_collection("my_docs")

collection.add(
    documents=docs,
    ids=[f"id{i}" for i in range(len(docs))]
)

results = collection.query(
    query_texts=["search query"],
    n_results=3
)
```

**You can do it! **
