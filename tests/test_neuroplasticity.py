import pytest
from unittest.mock import MagicMock, patch
from neuroform.memory.graph import KnowledgeGraph
from neuroform.memory.neuroplasticity import AutonomousNeuroplasticity

@pytest.fixture
def mock_kg():
    kg = MagicMock(spec=KnowledgeGraph)
    kg.driver = MagicMock()
    # Setup standard session return
    mock_session = MagicMock()
    kg.driver.session.return_value.__enter__.return_value = mock_session
    return kg

def test_evaluate_offline(mock_kg):
    mock_kg.driver = None
    neuro = AutonomousNeuroplasticity(mock_kg)
    res = neuro.evaluate_and_optimize()
    assert res["status"] == "offline"

def test_evaluate_empty_graph(mock_kg):
    mock_session = mock_kg.driver.session.return_value.__enter__.return_value
    mock_result = MagicMock()
    mock_result.__iter__.return_value = []
    mock_session.run.return_value = mock_result
    
    neuro = AutonomousNeuroplasticity(mock_kg)
    res = neuro.evaluate_and_optimize()
    assert res["status"] == "no_data"

@patch("neuroform.memory.neuroplasticity.ollama.chat")
def test_evaluate_with_actions(mock_chat, mock_kg):
    # Setup DB returning mock connection for fetch and proper mock for execute
    mock_session = mock_kg.driver.session.return_value.__enter__.return_value
    
    def run_side_effect(query, **kwargs):
        if "MATCH (a)-[r]->(b)" in query:
            return [{"source": "A", "relation": "KNOWS", "target": "B", "strength": 1.0}]
        
        mock_res = MagicMock()
        mock_res.consume.return_value.counters.properties_set = 0
        mock_res.peek.return_value = {"pruned": 0}
        return mock_res
        
    mock_session.run.side_effect = run_side_effect
    
    # Setup LLM returning JSON array of decisions
    llm_reply = '''```json
[
  {"action": "PRUNE", "source": "A", "relation": "KNOWS", "target": "B"},
  {"action": "STRENGTHEN", "source": "C", "relation": "LIKES", "target": "D"}
]
```'''
    mock_chat.return_value = {"message": {"content": llm_reply}}
    
    neuro = AutonomousNeuroplasticity(mock_kg)
    res = neuro.evaluate_and_optimize()
    
    assert res["status"] == "success"
    assert res["actions_taken"] == 2
    assert len(res["decisions"]) == 2
    
    # Verify execute calls
    assert mock_session.run.call_count > 1 # Fetch + Actions + Orphan cleanup

@patch("neuroform.memory.neuroplasticity.ollama.chat")
def test_evaluate_bad_json(mock_chat, mock_kg):
    mock_session = mock_kg.driver.session.return_value.__enter__.return_value
    
    def run_side_effect(query, **kwargs):
        if "MATCH (a)-[r]->(b)" in query:
            return [{"source": "X", "relation": "Y", "target": "Z", "strength": 1.0}]
            
        mock_res = MagicMock()
        mock_res.consume.return_value.counters.properties_set = 0
        mock_res.peek.return_value = {"pruned": 0}
        return mock_res
        
    mock_session.run.side_effect = run_side_effect
    
    mock_chat.return_value = {"message": {"content": "I think we shouldn't do anything"}}
    
    neuro = AutonomousNeuroplasticity(mock_kg)
    res = neuro.evaluate_and_optimize()
    
    assert res["status"] == "success"
    assert res["actions_taken"] == 0
    assert res["decisions"] == []

@patch("neuroform.memory.neuroplasticity.ollama.chat")
def test_evaluate_ollama_exception(mock_chat, mock_kg):
    mock_session = mock_kg.driver.session.return_value.__enter__.return_value
    mock_session.run.return_value = [{"source": "X", "relation": "Y", "target": "Z", "strength": 1.0}]
    
    mock_chat.side_effect = Exception("Model not found")
    
    neuro = AutonomousNeuroplasticity(mock_kg)
    res = neuro.evaluate_and_optimize()
    
    assert res["status"] == "error"
    assert "Model not found" in res["error"]

def test_execute_db_exception(mock_kg):
    neuro = AutonomousNeuroplasticity(mock_kg)
    
    # Force DB exception on execute, but we need it to trigger on the specific action run, not the cleanup run
    mock_session = mock_kg.driver.session.return_value.__enter__.return_value
    def run_side_effect(query, **kwargs):
        if "DELETE r" in query:
            raise Exception("Syntax error")
        return MagicMock()
        
    mock_session.run.side_effect = run_side_effect
    
    decisions = [{"action": "PRUNE", "source": "A", "relation": "R", "target": "B"}]
    
    # Should catch gracefully and return 0
    actions = neuro._execute_decisions(decisions)
    assert actions == 0

def test_execute_db_exception_cleanup(mock_kg):
    neuro = AutonomousNeuroplasticity(mock_kg)
    
    mock_session = mock_kg.driver.session.return_value.__enter__.return_value
    def run_side_effect(query, **kwargs):
        if "NOT (n)--()" in query:
            raise Exception("Cleanup error")
        return MagicMock()
        
    mock_session.run.side_effect = run_side_effect
    
    decisions = []
    
    # Should catch gracefully and return 0
    actions = neuro._execute_decisions(decisions)
    assert actions == 0

def test_execute_strengthen_and_decay(mock_kg):
    neuro = AutonomousNeuroplasticity(mock_kg)
    
    # We just want to make sure it doesn't crash and reaches the lines
    mock_session = mock_kg.driver.session.return_value.__enter__.return_value
    mock_session.run.return_value = MagicMock()
    
    decisions = [
        {"action": "STRENGTHEN", "source": "A", "relation": "R", "target": "B"},
        {"action": "DECAY", "source": "C", "relation": "Q", "target": "D"}
    ]
    
    actions = neuro._execute_decisions(decisions)
    assert actions == 2

def test_parse_llm_decisions_triple_backtick_no_lang():
    neuro = AutonomousNeuroplasticity(MagicMock())
    text = '```\n[{"action": "PRUNE"}]\n```'
    res = neuro._parse_llm_decisions(text)
    assert len(res) == 1

def test_parse_llm_decisions_not_a_list():
    neuro = AutonomousNeuroplasticity(MagicMock())
    text = '```json\n{"action": "PRUNE"}\n```'
    res = neuro._parse_llm_decisions(text)
    assert res == []

def test_apply_baseline_decay_no_driver():
    mock_kg = MagicMock()
    mock_kg.driver = None
    neuro = AutonomousNeuroplasticity(mock_kg)
    assert neuro.apply_baseline_decay() == 0

def test_apply_baseline_decay_exception():
    mock_kg = MagicMock()
    mock_session = mock_kg.driver.session.return_value.__enter__.return_value
    mock_session.run.side_effect = Exception("DB Error")
    
    neuro = AutonomousNeuroplasticity(mock_kg)
    # Exception should be caught and logged
    assert neuro.apply_baseline_decay() == 0

def test_apply_baseline_decay_success():
    mock_kg = MagicMock()
    mock_session = mock_kg.driver.session.return_value.__enter__.return_value
    
    def run_side_effect(query, **kwargs):
        res = MagicMock()
        if "SET s.strength" in query:
            res.consume.return_value.counters.properties_set = 5
        elif "DELETE n" in query:
            res.peek.return_value = {"pruned": 3}
        return res
        
    mock_session.run.side_effect = run_side_effect

    neuro = AutonomousNeuroplasticity(mock_kg)
    assert neuro.apply_baseline_decay() == 8

def test_execute_malformed_decision(mock_kg):
    neuro = AutonomousNeuroplasticity(mock_kg)
    decisions = [{"action": "PRUNE", "source": "A"}] # Missing relation, target
    actions = neuro._execute_decisions(decisions)
    assert actions == 0
