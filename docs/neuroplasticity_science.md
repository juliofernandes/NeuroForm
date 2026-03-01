# The Cognitive Science of NeuroForm

NeuroForm represents a departure from traditional, heuristically-driven database management. Instead of applying fixed mathematical formulas to decay or prune memory, NeuroForm mimics biological **neuroplasticity** by giving an Large Language Model (LLM) autonomous control over the semantic weights, structures, and lifecycle of a knowledge graph.

## What is Biological Neuroplasticity?

In cognitive science and neuroscience, "neuroplasticity" refers to the brain's ability to reorganize itself by forming new neural connections throughout life. 

This happens through a few core mechanisms:
1. **Hebbian Learning ("Neurons that fire together, wire together"):** When specific pathways are repeatedly activated, the synaptic strength between them increases.
2. **Synaptic Pruning:** Connections that are rarely used, contradictory, or rendered obsolete by new learning are weakened and eventually physically eliminated.
3. **Consolidation:** The process (often occurring during sleep/dreaming) where fragile, short-term experiences are filtered, restructured, and woven into stable long-term semantic memory.

## The Hybrid Computational Analogy

To model this, NeuroForm utilizes a **Dual-Phase Hybrid Engine**, combining deterministic baseline mathematics with higher-order LLM semantic reasoning:

### Phase 1: Baseline Homeostasis (The Mathematical Heuristic)
Biological synapses naturally decay over time without action potentials (use). In NeuroForm:
- **`last_fired` Timestamps:** Every time a node relationship is accessed through query context or successfully utilized, its `last_fired` timestamp is updated.
- **Mathematical Decay:** A recurring daemon applies a baseline decay rate (`-0.1` strength) to all non-structural connections based on their temporal inactivity.
- **Micro-Pruning:** Edges that fall below a strict threshold (`< 0.0`) are biologically purged by the system, mirroring the brain's automatic downscaling of sensory noise without needing conscious thought.

### Phase 2: Neuroplasticity (The Pre-frontal Semantic Layer)
While the heuristic layer handles raw temporal decay, the **Large Language Model (LLM)** acts as the cognitive reasoning center (the pre-frontal cortex) to evaluate the surviving edges.

The active graph slice is extracted and presented to the LLM, prompting it to act as an autonomous memory manager:

1.  **Semantic `STRENGTHEN` (Long-Term Potentiation):**
    If the LLM notices structural, critical facts (e.g., core system instructions, consistent user identity traits), it issues a `STRENGTHEN` command (`SET strength = ... + 0.5`). High-strength edges survive the baseline temporal decay much longer.
    
2.  **Semantic `DECAY` (Long-Term Depression):**
    If the LLM detects subjective memories losing relevance (e.g., "User was angry yesterday"), it issues a manual `DECAY` command to lower the strength (`- 0.2`). This accelerates the heuristic death of the node.
    
3.  **Contradiction `PRUNE` (Synaptic Elimination):**
    If the LLM identifies contradictory information (e.g., `(User)-[:LIKES]->(Coffee)` vs `(User)-[:HATES]->(Coffee)`), it acts to resolve cognitive dissonance by issuing a `PRUNE` command (`DELETE r`) on the outdated edge.

## Strengths of this Model

1. **Semantic Understanding of Obsolescence**: A math formula cannot tell that the fact `(User)-[LIVES_IN]->(Boston)` contradicts a newer fact `(User)-[LIVES_IN]->(New York)`. An LLM *can*. It can intelligently prune the Boston node because it understands the mutually exclusive semantics of residency, whereas a naive graph would keep both.
2. **Context Window Efficiency**: By pruning and decaying graph edges, NeuroForm keeps the ultimate context payload sent to the LLM during conversational turns lean and highly relevant, avoiding token-limit exhaustion.
3. **Graceful Forgetting**: It allows AI agents to "forget" gracefully over time, a vital feature for long-running autonomous agents dealing with changing human contexts.

## Cognitive Limits and Challenges

While powerful, this computational model has boundaries:

1. **Hallucination Cascades**: Because the LLM manages its *own* memory, a hallucinated consolidation judgment ("I should prune the user's name because it seems irrelevant") can permanently damage the graph's structural integrity. 
2. **Compute Cost of Introspection**: True Hebbian learning is cheap in biology but computationally expensive in silicon. Asking an LLM to actively evaluate thousands of graph nodes for pruning requires heavy background inference (which is why NeuroForm suggests using highly quantized, fast local models like standard `llama3`).
3. **Lack of Somatic Markers**: Biological consolidation relies heavily on emotional weight (the amygdala "tags" memories with emotional significance, prioritizing their survival). NeuroForm's LLM currently judges importance purely on text semantics, lacking the biological "feeling" that usually drives survival-critical memory consolidation.

## Summary

NeuroForm bridges the gap between rigid databases and fluid human memory. By trusting an LLM to autonomously execute `PRUNE`, `STRENGTHEN`, and `DECAY` operations on a Neo4j knowledge graph, it provides local AI agents with a dynamic, self-healing, and biologically analogous cognitive architecture.
