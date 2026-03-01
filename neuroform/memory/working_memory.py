"""
Working Memory — The Prefrontal Cortex Buffer
==============================================

Biological basis: The prefrontal cortex maintains a small (~4-7 item) active
workspace of current information. It gates what enters and exits attention,
preventing the brain from being overwhelmed by the full contents of long-term
memory.

Computational analogue: A fixed-size in-memory buffer (deque) that holds
recent conversation context and top graph retrievals. Before each LLM call,
the buffer ranks items by recency and strength to build an optimally focused
context window. Items that get referenced in the LLM's response are
automatically flagged for consolidation into long-term memory (Neo4j).
"""
import logging
from collections import deque
from typing import Dict, Any, List, Optional
import time

logger = logging.getLogger(__name__)


class WorkingMemoryItem:
    """A single item in working memory, with metadata for attention scoring."""

    def __init__(self, content: str, source: str = "conversation",
                 strength: float = 1.0, metadata: Optional[Dict[str, Any]] = None):
        self.content = content
        self.source = source  # "conversation", "graph", "system"
        self.strength = strength
        self.created_at = time.time()
        self.last_accessed = time.time()
        self.access_count = 0
        self.metadata = metadata or {}

    def access(self):
        """Mark this item as accessed (fires the neuron)."""
        self.last_accessed = time.time()
        self.access_count += 1

    def attention_score(self) -> float:
        """
        Score this item for attention gating.
        Combines recency, access frequency, and base strength.
        """
        age = time.time() - self.created_at
        recency_weight = 1.0 / (1.0 + age / 60.0)  # Decays over minutes
        frequency_weight = min(self.access_count / 3.0, 1.0)  # Caps at 3 accesses
        return (self.strength * 0.4) + (recency_weight * 0.4) + (frequency_weight * 0.2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "source": self.source,
            "strength": self.strength,
            "attention_score": self.attention_score(),
            "access_count": self.access_count,
            "metadata": self.metadata,
        }


class WorkingMemory:
    """
    The Prefrontal Cortex Buffer.

    A fixed-size in-memory workspace that maintains the most relevant
    context for the current cognitive task. Analogous to Miller's 7±2
    capacity limit of human working memory.
    """

    def __init__(self, capacity: int = 7):
        self.capacity = capacity
        self._buffer: deque = deque(maxlen=capacity)
        self._conversation_history: List[Dict[str, str]] = []
        self._max_history = 10  # Max conversation turns to keep

    @property
    def items(self) -> List[WorkingMemoryItem]:
        """Returns all items currently in working memory."""
        return list(self._buffer)

    @property
    def size(self) -> int:
        return len(self._buffer)

    def add(self, content: str, source: str = "conversation",
            strength: float = 1.0, metadata: Optional[Dict[str, Any]] = None) -> WorkingMemoryItem:
        """
        Add an item to working memory. If at capacity, the lowest-scoring
        item is automatically evicted (attentional displacement).
        """
        item = WorkingMemoryItem(content, source, strength, metadata)

        if len(self._buffer) >= self.capacity:
            # Evict the weakest item before adding
            self._evict_weakest()

        self._buffer.append(item)
        logger.debug(f"WM: Added [{source}] item (size: {len(self._buffer)}/{self.capacity})")
        return item

    def add_conversation_turn(self, role: str, content: str):
        """Add a conversation turn to the rolling history."""
        self._conversation_history.append({"role": role, "content": content})
        if len(self._conversation_history) > self._max_history:
            self._conversation_history = self._conversation_history[-self._max_history:]

        # Also add to the working memory buffer for attention scoring
        self.add(content, source="conversation", strength=1.0,
                 metadata={"role": role})

    def add_graph_context(self, context_items: List[Dict[str, Any]]):
        """
        Inject graph query results into working memory.
        Each context item is expected to have 'source', 'relationship',
        'target', and 'strength' keys.
        """
        for item in context_items:
            content = f"{item.get('source', '?')} ({item.get('relationship', '?')}) {item.get('target', '?')}"
            self.add(
                content=content,
                source="graph",
                strength=item.get("strength", 1.0),
                metadata=item
            )

    def attend(self, top_k: Optional[int] = None) -> List[WorkingMemoryItem]:
        """
        Attention gating: returns items ranked by attention score.
        Marks accessed items as 'fired'. Optionally limits to top_k.
        """
        ranked = sorted(self._buffer, key=lambda x: x.attention_score(), reverse=True)

        if top_k is not None:
            ranked = ranked[:top_k]

        for item in ranked:
            item.access()

        return ranked

    def build_context_string(self, top_k: Optional[int] = None) -> str:
        """
        Build a formatted context string from the most relevant items
        in working memory, suitable for injection into an LLM prompt.
        """
        attended = self.attend(top_k=top_k)

        if not attended:
            return "No prior memory context."

        parts = ["Prior Memory:"]
        for item in attended:
            parts.append(f"- [{item.source.upper()}] {item.content}")

        return "\n".join(parts)

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Return the rolling conversation history for multi-turn prompts."""
        return list(self._conversation_history)

    def _evict_weakest(self):
        """Remove the lowest-scoring item from the buffer."""
        if not self._buffer:
            return

        weakest = min(self._buffer, key=lambda x: x.attention_score())
        self._buffer.remove(weakest)
        logger.debug(f"WM: Evicted [{weakest.source}] item (score: {weakest.attention_score():.3f})")

    def clear(self):
        """Wipe working memory and conversation history."""
        self._buffer.clear()
        self._conversation_history.clear()

    def get_items_by_source(self, source: str) -> List[WorkingMemoryItem]:
        """Filter buffer contents by source type."""
        return [item for item in self._buffer if item.source == source]

    def snapshot(self) -> Dict[str, Any]:
        """Return a diagnostic snapshot of the current working memory state."""
        return {
            "capacity": self.capacity,
            "size": len(self._buffer),
            "items": [item.to_dict() for item in self._buffer],
            "conversation_turns": len(self._conversation_history),
        }
