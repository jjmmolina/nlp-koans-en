# Theory: Modern LLMs

## What are Modern LLMs?

Large Language Models trained on massive datasets with billions of parameters.

## Evolution

- **GPT-2** (2019): 1.5B parameters
- **GPT-3** (2020): 175B parameters
- **GPT-4** (2023): Multimodal
- **Claude, Llama, Gemini**: Competition era

## Key Features

### Scale
More parameters = better performance (usually)

### Few-Shot Learning
Learn from examples in prompt without retraining.

```python
prompt = """
Translate to French:
English: Hello
French: Bonjour

English: Goodbye
French: Au revoir

English: Thank you
French:"""
```

### Emergent Abilities
New capabilities appear at scale:
- Reasoning
- Arithmetic
- Code generation

## Using LLMs

### OpenAI API
```python
from openai import OpenAI

client = OpenAI(api_key="YOUR_KEY")
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Hugging Face
```python
from transformers import pipeline

llm = pipeline("text-generation", model="meta-llama/Llama-2-7b-chat-hf")
result = llm("Explain quantum computing")
```

### LangChain
Framework for building LLM applications.

## Challenges

- **Hallucinations**: Making up facts
- **Bias**: Reflecting training data biases
- **Cost**: API calls can be expensive
- **Privacy**: Sensitive data concerns

## Applications

- Chatbots and assistants
- Code generation
- Content creation
- Research and analysis

**Practice with tests! **
