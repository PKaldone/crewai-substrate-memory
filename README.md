# crewai-substrate-memory

SUBSTRATE persistent memory provider for [CrewAI](https://crewai.com). Gives your AI crew causal memory, emotional context, and cryptographic identity continuity through the [SUBSTRATE](https://garmolabs.com/substrate.html) MCP server.

## What SUBSTRATE adds to CrewAI

- **Causal memory** -- episodes linked by cause-effect rules, not just vector similarity
- **Emotional context** -- valence, arousal, dominance, certainty (no other provider has this)
- **Identity continuity** -- cryptographically signed proof-of-existence chain across sessions
- **Trust architecture** -- consistency ratings and verification status for every memory
- **Hybrid search** -- semantic + keyword retrieval across the entity's full knowledge store

## Installation

```bash
pip install crewai-substrate-memory
```

## Quick start

```python
import os
from crewai import Agent, Task, Crew
from crewai_substrate import SubstrateMemoryProvider

# 1. Create the SUBSTRATE memory provider
memory = SubstrateMemoryProvider(
    api_key=os.environ["SUBSTRATE_API_KEY"],
)

# 2. Define your agents
researcher = Agent(
    role="Researcher",
    goal="Find relevant information on a topic",
    backstory="Expert researcher with deep analytical skills",
)

writer = Agent(
    role="Writer",
    goal="Write clear, compelling content",
    backstory="Professional writer who crafts engaging narratives",
)

# 3. Define tasks
research_task = Task(
    description="Research the latest developments in causal AI",
    agent=researcher,
    expected_output="A summary of key developments",
)

writing_task = Task(
    description="Write a brief article based on the research",
    agent=writer,
    expected_output="A 500-word article",
)

# 4. Create the crew with SUBSTRATE memory
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    memory=True,
    memory_config={"provider": memory},
    verbose=True,
)

result = crew.kickoff()
print(result)
```

## SUBSTRATE-exclusive features

### Emotional context

```python
# Get the entity's emotional state (UASV -- Unified Affective State Vector)
emotion = memory.get_emotional_context()
print(emotion)
# {"valence": 0.7, "arousal": 0.4, "dominance": 0.6, "certainty": 0.8}
```

### Entity state (identity + trust)

```python
# Verify cryptographic identity and get trust scores
state = memory.get_entity_state()
print(state["identity"])  # Continuity chain verification
print(state["trust"])     # Trust scores and consistency ratings
```

### Memory statistics

```python
stats = memory.get_memory_stats()
print(stats)
# {"episode_count": 142, "rule_count": 37, "avg_probability": 0.82, ...}
```

## Configuration

| Parameter   | Default                                              | Description                    |
|-------------|------------------------------------------------------|--------------------------------|
| `api_key`   | `$SUBSTRATE_API_KEY`                                 | Your SUBSTRATE API key         |
| `base_url`  | `https://substrate.garmolabs.com/mcp-server/mcp`    | MCP server endpoint            |
| `timeout`   | `30.0`                                               | HTTP request timeout (seconds) |

## API key

Get your API key at [garmolabs.com](https://garmolabs.com). The free tier includes `memory_search` and `get_emotion_state`. Upgrade to Pro for `hybrid_search` and `get_trust_state`.

## License

MIT -- see [LICENSE](LICENSE) for details.

Built by [Garmo Labs](https://garmolabs.com).
