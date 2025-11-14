# Hints: AI Agents

## Common Issues

### Agent loops infinitely
- Set max_iterations limit
- Improve tool descriptions
- Better initial prompt

### Tools not being called
- Make tool descriptions clearer
- Check tool is registered correctly
- Verify LLM supports function calling

### Expensive API calls
- Set iteration limits
- Use cheaper models for testing
- Cache results

## Quick Solutions

**Basic LangChain agent:**
```python
from langchain.agents import initialize_agent, load_tools
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)
tools = load_tools(["llm-math"], llm=llm)
agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True,
    max_iterations=5
)
result = agent.run("What is 25 * 4?")
```

**Custom tool:**
```python
from langchain.tools import Tool

def my_function(query: str) -> str:
    return f"Result: {query}"

tool = Tool(
    name="MyTool",
    func=my_function,
    description="Useful for doing X. Input should be Y."
)
```

**ReAct pattern:**
1. Thought: Reason about what to do
2. Action: Call a tool
3. Observation: See result
4. Repeat until answer found

**You can do it! **
