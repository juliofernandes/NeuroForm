"""
Unit Tests for WorkingMemory (Prefrontal Cortex Buffer)
=======================================================
100% coverage of WorkingMemoryItem and WorkingMemory classes.
"""
import pytest
import time
from unittest.mock import patch
from neuroform.memory.working_memory import WorkingMemory, WorkingMemoryItem


# ===========================================================================
# WorkingMemoryItem Tests
# ===========================================================================
class TestWorkingMemoryItem:

    def test_creation_defaults(self):
        item = WorkingMemoryItem("hello world")
        assert item.content == "hello world"
        assert item.source == "conversation"
        assert item.strength == 1.0
        assert item.access_count == 0
        assert item.metadata == {}
        assert item.created_at > 0
        assert item.last_accessed > 0

    def test_creation_with_params(self):
        item = WorkingMemoryItem("fact", source="graph", strength=3.0, metadata={"key": "val"})
        assert item.source == "graph"
        assert item.strength == 3.0
        assert item.metadata == {"key": "val"}

    def test_access_increments_count(self):
        item = WorkingMemoryItem("test")
        assert item.access_count == 0
        item.access()
        assert item.access_count == 1
        item.access()
        assert item.access_count == 2

    def test_access_updates_last_accessed(self):
        item = WorkingMemoryItem("test")
        old_accessed = item.last_accessed
        time.sleep(0.01)
        item.access()
        assert item.last_accessed >= old_accessed

    def test_attention_score_components(self):
        item = WorkingMemoryItem("test", strength=2.0)
        score = item.attention_score()
        # Should be positive and reasonable
        assert score > 0
        assert isinstance(score, float)

    def test_attention_score_increases_with_strength(self):
        weak = WorkingMemoryItem("a", strength=0.1)
        strong = WorkingMemoryItem("b", strength=5.0)
        assert strong.attention_score() > weak.attention_score()

    def test_attention_score_frequency_cap(self):
        item = WorkingMemoryItem("test")
        # Access many times (frequency caps at 3)
        for _ in range(10):
            item.access()
        score = item.attention_score()
        assert score > 0

    def test_to_dict(self):
        item = WorkingMemoryItem("fact", source="graph", strength=2.0, metadata={"k": "v"})
        d = item.to_dict()
        assert d["content"] == "fact"
        assert d["source"] == "graph"
        assert d["strength"] == 2.0
        assert d["metadata"] == {"k": "v"}
        assert "attention_score" in d
        assert d["access_count"] == 0


# ===========================================================================
# WorkingMemory Tests
# ===========================================================================
class TestWorkingMemory:

    def test_init_defaults(self):
        wm = WorkingMemory()
        assert wm.capacity == 7
        assert wm.size == 0
        assert wm.items == []

    def test_init_custom_capacity(self):
        wm = WorkingMemory(capacity=3)
        assert wm.capacity == 3

    def test_add_item(self):
        wm = WorkingMemory()
        item = wm.add("hello", source="conversation")
        assert wm.size == 1
        assert isinstance(item, WorkingMemoryItem)
        assert item.content == "hello"

    def test_add_returns_item(self):
        wm = WorkingMemory()
        item = wm.add("test", source="graph", strength=2.0, metadata={"a": 1})
        assert item.source == "graph"
        assert item.strength == 2.0
        assert item.metadata == {"a": 1}

    def test_capacity_eviction(self):
        wm = WorkingMemory(capacity=3)
        wm.add("a")
        wm.add("b")
        wm.add("c")
        assert wm.size == 3
        # Adding a 4th should evict the weakest
        wm.add("d")
        assert wm.size == 3
        contents = [item.content for item in wm.items]
        assert "d" in contents

    def test_evict_weakest_removes_lowest_score(self):
        wm = WorkingMemory(capacity=3)
        wm.add("strong", strength=5.0)
        wm.add("medium", strength=2.0)
        wm.add("weak", strength=0.1)
        wm.add("new", strength=1.0)
        # "weak" should be evicted
        contents = [item.content for item in wm.items]
        assert "strong" in contents
        assert "new" in contents

    def test_evict_weakest_empty_buffer(self):
        wm = WorkingMemory()
        # Should not crash
        wm._evict_weakest()
        assert wm.size == 0

    def test_add_conversation_turn(self):
        wm = WorkingMemory()
        wm.add_conversation_turn("user", "hello")
        assert wm.size == 1
        history = wm.get_conversation_history()
        assert len(history) == 1
        assert history[0] == {"role": "user", "content": "hello"}

    def test_conversation_history_truncation(self):
        wm = WorkingMemory(capacity=20)
        wm._max_history = 3
        for i in range(5):
            wm.add_conversation_turn("user", f"msg {i}")
        history = wm.get_conversation_history()
        assert len(history) == 3
        assert history[0]["content"] == "msg 2"

    def test_add_graph_context(self):
        wm = WorkingMemory()
        context = [
            {"source": "User", "relationship": "LIKES", "target": "Python", "strength": 3.0},
            {"source": "User", "relationship": "KNOWS", "target": "Alice", "strength": 1.0},
        ]
        wm.add_graph_context(context)
        assert wm.size == 2
        graph_items = wm.get_items_by_source("graph")
        assert len(graph_items) == 2

    def test_add_graph_context_missing_fields(self):
        wm = WorkingMemory()
        context = [{"other": "data"}]
        wm.add_graph_context(context)
        assert wm.size == 1
        assert "?" in wm.items[0].content

    def test_attend_returns_ranked(self):
        wm = WorkingMemory()
        wm.add("weak", strength=0.1)
        wm.add("strong", strength=5.0)
        ranked = wm.attend()
        assert ranked[0].content == "strong"

    def test_attend_top_k(self):
        wm = WorkingMemory()
        wm.add("a")
        wm.add("b")
        wm.add("c")
        ranked = wm.attend(top_k=2)
        assert len(ranked) == 2

    def test_attend_marks_accessed(self):
        wm = WorkingMemory()
        wm.add("test")
        wm.attend()
        assert wm.items[0].access_count == 1

    def test_build_context_string_empty(self):
        wm = WorkingMemory()
        ctx = wm.build_context_string()
        assert ctx == "No prior memory context."

    def test_build_context_string_with_items(self):
        wm = WorkingMemory()
        wm.add("fact A", source="graph")
        wm.add("hello user", source="conversation")
        ctx = wm.build_context_string()
        assert "Prior Memory:" in ctx
        assert "[GRAPH]" in ctx
        assert "[CONVERSATION]" in ctx

    def test_build_context_string_top_k(self):
        wm = WorkingMemory()
        for i in range(5):
            wm.add(f"item {i}")
        ctx = wm.build_context_string(top_k=2)
        # Should have header + 2 items
        lines = [l for l in ctx.split("\n") if l.strip()]
        assert len(lines) == 3

    def test_get_conversation_history_is_copy(self):
        wm = WorkingMemory()
        wm.add_conversation_turn("user", "hi")
        h = wm.get_conversation_history()
        h.clear()
        assert len(wm.get_conversation_history()) == 1

    def test_clear(self):
        wm = WorkingMemory()
        wm.add("a")
        wm.add_conversation_turn("user", "b")
        wm.clear()
        assert wm.size == 0
        assert len(wm.get_conversation_history()) == 0

    def test_get_items_by_source(self):
        wm = WorkingMemory()
        wm.add("a", source="graph")
        wm.add("b", source="conversation")
        wm.add("c", source="graph")
        graphs = wm.get_items_by_source("graph")
        assert len(graphs) == 2
        convos = wm.get_items_by_source("conversation")
        assert len(convos) == 1
        assert wm.get_items_by_source("system") == []

    def test_snapshot(self):
        wm = WorkingMemory(capacity=5)
        wm.add("test_item")
        wm.add_conversation_turn("user", "hi")
        snap = wm.snapshot()
        assert snap["capacity"] == 5
        assert snap["size"] == 2
        assert len(snap["items"]) == 2
        assert snap["conversation_turns"] == 1

    def test_items_property_is_copy(self):
        wm = WorkingMemory()
        wm.add("x")
        items = wm.items
        items.clear()
        assert wm.size == 1  # Original unaffected
