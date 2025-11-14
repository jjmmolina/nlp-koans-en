# Hints: Modern LLMs

## Common Issues

### API key errors
Set environment variable or pass directly:
```python
import os
os.environ["OPENAI_API_KEY"] = "your-key"
```

### Rate limits
- Add delays between calls
- Use smaller models
- Batch requests

### High costs
- Use caching
- Start with smaller/cheaper models
- Set max_tokens limit

### Model not available
Check model name and your API access level.

## Quick Solutions

**OpenAI:**
```python
from openai import OpenAI

client = OpenAI(api_key="YOUR_KEY")
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello!"}]
)
text = response.choices[0].message.content
```

**HuggingFace:**
```python
from transformers import pipeline

llm = pipeline("text-generation", model="gpt2")
result = llm("Hello", max_length=50)
```

**LangChain:**
```python
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

llm = OpenAI(temperature=0.7)
prompt = PromptTemplate.from_template("Tell me about {topic}")
chain = prompt | llm
result = chain.invoke({"topic": "NLP"})
```

**You can do it! **
