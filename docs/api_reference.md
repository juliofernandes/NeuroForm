# NeuroForm API Reference

## `neuroform.memory.graph`

### `KnowledgeGraph`

The core Neo4j interaction class. Defaults to looking for environment variables: `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`.

- **`connect()`**: Establishes the driver connection. Called automatically on initialization. Checks `DISABLE_NEO4J` environment variable for graceful degradation.
- **`close()`**: Safely closes the Neo4j driver session.
- **`clear_all()`**: **DANGER**. Truncates the entire graph. Used primarily for testing.
- **`add_node(label: str, name: str, layer: str, properties: dict = None)`**: Merges a node into the graph.
- **`add_relationship(source_name: str, rel_type: str, target_name: str, strength: float = 1.0)`**: Creates or updates an edge. Automatically sanitizes `rel_type` to Neo4j-compliant uppercase strings. Sets or updates the `last_fired` timestamp.
- **`query_context(entity_name: str, layer: str = None, limit: int = 10)`**: Retrieves surrounding nodes from a given entity, sorted descending by relationship strength. Updates the `last_fired` timestamp for all returned edges to simulate biological action potentials.

---

## `neuroform.memory.neuroplasticity`

### `AutonomousNeuroplasticity`

The LLM-driven graph optimization daemon.

- **`__init__(kg: KnowledgeGraph, model: str = "llama3")`**: Initializes the optimizer with a reference to the graph and an Ollama model string to use for background reasoning.
- **`apply_baseline_decay(decay_rate: float = 0.1, prune_threshold: float = 0.0) -> int`**: Executes the mathematical heuristic stage. Subtracts `decay_rate` from all non-structural connections and completely drops any that fall below `prune_threshold`. Deletes any resulting orphan nodes mathematically.
- **`evaluate_and_optimize() -> dict`**: The main execution loop. It first calls `apply_baseline_decay()`. Then it fetches a slice of the graph, asks the LLM to evaluate it, parses the JSON response, and executes the semantic `PRUNE`, `STRENGTHEN`, and `DECAY` Cypher operations. Returns a dict containing the status, number of heuristic + LLM actions taken, and raw decision array.

---

## `neuroform.llm.ollama_client`

### `OllamaClient`

The user-facing chat layer.

- **`__init__(kg: KnowledgeGraph, model: str = "llama3")`**: Initializes the client.
- **`chat_with_memory(user_id: str, message: str) -> str`**: Processes a chat turn. Pulls context, sends the system and user prompts to Ollama, strips internal JSON blocks from the output, saves the extracted JSON facts to the graph, and returns the cleaned text response to the user.
