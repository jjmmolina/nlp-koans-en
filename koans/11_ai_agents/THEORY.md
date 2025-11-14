# Theory: AI Agents

## What are AI Agents?

Autonomous systems that use LLMs to make decisions and take actions.

## Components

### 1. LLM Brain
The reasoning engine (GPT-4, Claude, etc.)

### 2. Tools
Functions the agent can call:
- Web search
- Calculator
- Database queries
- API calls

### 3. Memory
Stores conversation history and context

### 4. Planning
Breaks complex tasks into steps

## How Agents Work

```
User: "What's the weather in Paris and convert temperature to Fahrenheit"

Agent:
1. Call weather_api("Paris")  20C
2. Call calculator(20 * 9/5 + 32)  68F
3. Respond: "68F in Paris"
```

## Frameworks

### LangChain
```python
from langchain.agents import initialize_agent
from langchain.agents import load_tools

tools = load_tools(["serpapi", "llm-math"])
agent = initialize_agent(tools, llm, agent="zero-shot-react-description")
agent.run("What's the GDP of France in 2023?")
```

### AutoGPT
Autonomous agent that creates its own goals.

### ReAct Pattern
**Re**asoning + **Act**ing loop:
1. Thought: What should I do?
2. Action: Use tool
3. Observation: See result
4. Repeat until done

## Types of Agents

- **Conversational**: Chat with context
- **Task-oriented**: Complete specific goals
- **Autonomous**: Self-directed
- **Collaborative**: Multiple agents

## Challenges

- **Reliability**: Agents can fail or loop
- **Cost**: Many LLM calls
- **Safety**: Need guardrails
- **Debugging**: Hard to trace decisions

**Practice with tests! **
