"""
Unit Tests for DreamConsolidation — Hippocampal Replay
======================================================
100% coverage of the DreamConsolidation class.
"""
import pytest
from unittest.mock import MagicMock, patch
from neuroform.memory.graph import KnowledgeGraph, GraphLayer
from neuroform.memory.dream_consolidation import DreamConsolidation


@pytest.fixture
def mock_kg():
    kg = MagicMock(spec=KnowledgeGraph)
    kg.driver = MagicMock()
    return kg


@pytest.fixture
def mock_kg_offline():
    kg = MagicMock(spec=KnowledgeGraph)
    kg.driver = None
    return kg


# ===========================================================================
# consolidate() — Main Entry Point
# ===========================================================================
class TestConsolidate:

    def test_offline_returns_status(self, mock_kg_offline):
        dc = DreamConsolidation(mock_kg_offline)
        result = dc.consolidate()
        assert result["status"] == "offline"
        assert result["episodes_processed"] == 0

    @patch("neuroform.memory.dream_consolidation.ollama.chat")
    def test_no_episodes_returns_early(self, mock_chat, mock_kg):
        session = MagicMock()
        session.run.return_value = iter([])  # No episodes
        mock_kg.driver.session.return_value.__enter__ = MagicMock(return_value=session)
        mock_kg.driver.session.return_value.__exit__ = MagicMock(return_value=False)

        dc = DreamConsolidation(mock_kg)
        result = dc.consolidate()
        assert result["status"] == "no_episodes"
        mock_chat.assert_not_called()

    @patch("neuroform.memory.dream_consolidation.ollama.chat")
    def test_full_consolidation_cycle(self, mock_chat, mock_kg):
        # Mock: episodes returned from graph
        episode_record = {
            "source": "User", "relation": "VISITED", "target": "Coffee Shop",
            "strength": 1.0
        }
        session = MagicMock()
        
        # First call (_fetch_recent_episodes) returns episodes
        episode_result = MagicMock()
        episode_result.__iter__ = MagicMock(return_value=iter([episode_record]))
        
        # Decay call returns nothing special
        decay_result = MagicMock()
        
        session.run.side_effect = [episode_result, decay_result]
        mock_kg.driver.session.return_value.__enter__ = MagicMock(return_value=session)
        mock_kg.driver.session.return_value.__exit__ = MagicMock(return_value=False)

        # Mock LLM response: distilled semantic facts
        mock_chat.return_value = {"message": {"content": """```json
[{"source": "User", "relation": "PREFERS", "target": "Coffee", "confidence": 0.9}]
```"""}}

        dc = DreamConsolidation(mock_kg)
        result = dc.consolidate()

        assert result["status"] == "consolidated"
        assert result["episodes_processed"] == 1
        assert result["semantics_created"] == 1
        assert result["episodes_decayed"] == 1

        # Verify semantic nodes were written
        mock_kg.add_node.assert_called()
        mock_kg.add_relationship.assert_called()


# ===========================================================================
# _fetch_recent_episodes
# ===========================================================================
class TestFetchRecentEpisodes:

    def test_returns_list_of_dicts(self, mock_kg):
        session = MagicMock()
        record = {"source": "User", "relation": "ATE", "target": "Pizza", "strength": 2.0}
        session.run.return_value = iter([record])
        mock_kg.driver.session.return_value.__enter__ = MagicMock(return_value=session)
        mock_kg.driver.session.return_value.__exit__ = MagicMock(return_value=False)

        dc = DreamConsolidation(mock_kg)
        episodes = dc._fetch_recent_episodes(21_600_000)
        assert len(episodes) == 1
        assert episodes[0]["source"] == "User"


# ===========================================================================
# _distill_episodes
# ===========================================================================
class TestDistillEpisodes:

    @patch("neuroform.memory.dream_consolidation.ollama.chat")
    def test_parses_json_response(self, mock_chat, mock_kg):
        mock_chat.return_value = {"message": {"content": """```json
[{"source": "User", "relation": "LIKES", "target": "Tea"}]
```"""}}
        dc = DreamConsolidation(mock_kg)
        result = dc._distill_episodes([{"source": "User", "relation": "DRANK", "target": "Tea", "strength": 1.0}])
        assert len(result) == 1
        assert result[0]["target"] == "Tea"

    @patch("neuroform.memory.dream_consolidation.ollama.chat")
    def test_llm_exception_returns_empty(self, mock_chat, mock_kg):
        mock_chat.side_effect = Exception("LLM down")
        dc = DreamConsolidation(mock_kg)
        result = dc._distill_episodes([{"source": "User", "relation": "SAW", "target": "Cat", "strength": 1.0}])
        assert result == []


# ===========================================================================
# _parse_semantics
# ===========================================================================
class TestParseSemantics:

    def test_parse_with_json_block(self, mock_kg):
        dc = DreamConsolidation(mock_kg)
        text = """Here are the results:
```json
[{"source": "A", "relation": "R", "target": "B"}]
```"""
        result = dc._parse_semantics(text)
        assert len(result) == 1

    def test_parse_with_generic_code_block(self, mock_kg):
        dc = DreamConsolidation(mock_kg)
        text = """```
[{"source": "A", "relation": "R", "target": "B"}]
```"""
        result = dc._parse_semantics(text)
        assert len(result) == 1

    def test_parse_raw_json(self, mock_kg):
        dc = DreamConsolidation(mock_kg)
        text = '[{"source": "A", "relation": "R", "target": "B"}]'
        result = dc._parse_semantics(text)
        assert len(result) == 1

    def test_parse_bad_json(self, mock_kg):
        dc = DreamConsolidation(mock_kg)
        result = dc._parse_semantics("not json at all")
        assert result == []

    def test_parse_non_list_json(self, mock_kg):
        dc = DreamConsolidation(mock_kg)
        result = dc._parse_semantics('{"key": "value"}')
        assert result == []


# ===========================================================================
# _write_semantic_nodes
# ===========================================================================
class TestWriteSemanticNodes:

    def test_writes_valid_facts(self, mock_kg):
        dc = DreamConsolidation(mock_kg)
        facts = [{"source": "User", "relation": "LIKES", "target": "Music"}]
        created = dc._write_semantic_nodes(facts)
        assert created == 1
        assert mock_kg.add_node.call_count == 2
        mock_kg.add_relationship.assert_called_once()

    def test_skips_incomplete_facts(self, mock_kg):
        dc = DreamConsolidation(mock_kg)
        facts = [{"source": "User"}]  # Missing relation and target
        created = dc._write_semantic_nodes(facts)
        assert created == 0
        mock_kg.add_node.assert_not_called()

    def test_handles_db_exception(self, mock_kg):
        mock_kg.add_node.side_effect = Exception("DB error")
        dc = DreamConsolidation(mock_kg)
        facts = [{"source": "User", "relation": "LIKES", "target": "Music"}]
        created = dc._write_semantic_nodes(facts)
        assert created == 0


# ===========================================================================
# _decay_episodes
# ===========================================================================
class TestDecayEpisodes:

    def test_decays_unique_sources(self, mock_kg):
        session = MagicMock()
        mock_kg.driver.session.return_value.__enter__ = MagicMock(return_value=session)
        mock_kg.driver.session.return_value.__exit__ = MagicMock(return_value=False)

        dc = DreamConsolidation(mock_kg)
        episodes = [
            {"source": "User", "relation": "DID", "target": "A", "strength": 1.0},
            {"source": "User", "relation": "DID", "target": "B", "strength": 1.0},
            {"source": "Event1", "relation": "HAD", "target": "C", "strength": 1.0},
        ]
        decayed = dc._decay_episodes(episodes)
        assert decayed == 2  # "User" and "Event1" (unique sources)
        assert session.run.call_count == 2

    def test_handles_decay_exception(self, mock_kg):
        session = MagicMock()
        session.run.side_effect = Exception("DB error")
        mock_kg.driver.session.return_value.__enter__ = MagicMock(return_value=session)
        mock_kg.driver.session.return_value.__exit__ = MagicMock(return_value=False)

        dc = DreamConsolidation(mock_kg)
        episodes = [{"source": "A", "relation": "R", "target": "B", "strength": 1.0}]
        decayed = dc._decay_episodes(episodes)
        assert decayed == 0
