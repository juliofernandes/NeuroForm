# NeuroForm Architectural Overview

NeuroForm is built on three core pillars: structured graph storage (Neo4j), conversational intelligence (Ollama), and an autonomous optimization daemon loop.

This document outlines the technical flow between these components.

## 1. Interaction Layer (`OllamaClient`)

The interaction layer, `OllamaClient` (`neuroform.llm.ollama_client`), is responsible for user-facing chatting and immediate memory extraction.

**The Flow:**
1. **Context Retrieval**: When a user sends a message, the client queries `KnowledgeGraph` for any relevant nodes and edges connected to the user or the topic, sorted by edge `strength`.
2. **Prompt Injection**: This graph context is injected into the LLM system prompt under a `[SRC:KG]` header.
3. **Inference**: Ollama generates a response based on its weights *and* the injected graph context.
4. **Autonomous Fact Extraction**: The LLM is instructed in the system prompt to append a structured ````json` block to its reply *if* it learns a new, long-term fact. The client strips this block from the user's view, parses it, and immediately pushes the new nodes/edges to the `KnowledgeGraph`.

## 2. Storage Layer (`KnowledgeGraph`)

The `KnowledgeGraph` (`neuroform.memory.graph`) acts as the Neo4j driver wrapper.

**Key responsibilities:**
- **Schema Initialization**: On boot, it ensures indexes exist for high-speed node lookups (e.g., indexing on `name` and cognitive `layer`).
- **CRUD Abstraction**: Exposes simple python methods (`add_node`, `add_relationship`, `query_context`) that generate sanitized Cypher queries. 
- **Edge Weighting**: All relationships have a default `strength` of 1.0. If `add_relationship` is called on an existing edge, the `strength` is incremented.

## 3. Autonomous Daemon (`AutonomousNeuroplasticity`)

This is the system's "subconscious", implemented in `neuroform.memory.neuroplasticity`. It runs the `evaluate_and_optimize()` loop.

**The Dual-Phase Flow:**
1. **Mathematical Heuristic Phase (Baseline Homeostasis)**: 
   - Before hitting the LLM, the system runs a fast Cypher query (`apply_baseline_decay()`). It universally reduces the strength of all non-structural ties (simulating biological time passing) and auto-deletes any node that falls below 0, followed by a cleanup of orphaned sub-nodes.
2. **Graph Sampling (Semantic Phase)**: 
   - The daemon pulls a slice of the surviving active graph (nodes, relationships, and their current strengths).
3. **Introspection Prompt**: 
   - It feeds this pure JSON graph slice to the LLM with a system prompt asking the model to act as a pre-frontal memory manager.
3. **Reasoning**: The LLM evaluates the slice for:
   - **Contradictions / Obsolescence**: (e.g., "User likes Pizza" vs "User is allergic to dairy").
   - **Structural Importance**: Core facts that should never fade.
   - **Irrelevance**: Ephemeral facts that no longer matter.
4. **Action Parsing**: The LLM returns a JSON array of specific actions:
   - `{"action": "PRUNE", "source": "A", "relation": "R", "target": "B"}`
   - `{"action": "STRENGTHEN", ...}`
   - `{"action": "DECAY", ...}`
6. **Cypher Execution**: The daemon parses this JSON array and executes direct `DELETE` or `SET strength = ...` Cypher queries against the Neo4j backend.

## 6-Tier Architecture & Layer Topology (ErnOS Style)

NeuroForm implements a dynamic root-node layer topology, identical to the architecture found in ErnOS.

**Layer Roots:**
Every time a node is created in a specific layer (e.g., `EPISODIC`, `SEMANTIC`), the graph ensures a `LayerRoot` node exists for that layer (e.g., `(:LayerRoot {name: 'EPISODIC'})`).
- **Interconnected Roots**: All `LayerRoot` nodes are connected to each other via a `[:PEER_LAYER]` edge, creating a full mesh at the top level of the graph.
- **Node Attachment**: Every standard entity node added to a layer is attached directly to its corresponding `LayerRoot` via an `[:IN_LAYER]` edge.

While the LLM can dynamically spawn *new* layers on the fly by dictating custom strings, NeuroForm supports the core ErnOS 6-tier cognitive architecture out of the box:

1. `NARRATIVE`: General conversation history.
2. `SEMANTIC`: Definitions and hard facts.
3. `EPISODIC`: Time-bound events.
4. `SOCIAL`: Relationship graphs.
5. `SYSTEM`: Metadata and formatting.
6. `PROCEDURAL`: Skills and "how-to" knowledge. 

NeuroForm provides the `GraphLayer` enum in `graph.py` to facilitate this structural tagging if desired.
