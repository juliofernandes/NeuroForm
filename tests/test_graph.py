import pytest
import os
from unittest.mock import MagicMock
from neuroform.memory.graph import KnowledgeGraph, GraphLayer

@pytest.fixture
def mock_neo4j(mocker):
    # Mock GraphDatabase to avoid actual network calls
    mock_db = mocker.patch("neuroform.memory.graph.GraphDatabase")
    mock_driver = MagicMock()
    mock_db.driver.return_value = mock_driver
    return mock_db, mock_driver

@pytest.fixture
def clean_env():
    # Ensure env vars are clean for tests
    old_env = dict(os.environ)
    if "DISABLE_NEO4J" in os.environ:
        del os.environ["DISABLE_NEO4J"]
    yield
    os.environ.clear()
    os.environ.update(old_env)

def test_init_connects_successfully(mock_neo4j, clean_env):
    mock_db, mock_driver = mock_neo4j
    kg = KnowledgeGraph(uri="bolt://test:7687", user="test", password="test")
    
    mock_db.driver.assert_called_once_with("bolt://test:7687", auth=("test", "test"))
    mock_driver.verify_connectivity.assert_called_once()
    assert kg.driver == mock_driver

def test_init_fails_gracefully(mock_neo4j, clean_env):
    mock_db, mock_driver = mock_neo4j
    mock_driver.verify_connectivity.side_effect = Exception("Connection Failed")
    
    kg = KnowledgeGraph()
    # Driver should be None if connection failed
    assert kg.driver is None

def test_init_disabled_via_env(mock_neo4j):
    mock_db, mock_driver = mock_neo4j
    os.environ["DISABLE_NEO4J"] = "true"
    
    kg = KnowledgeGraph()
    assert kg.driver is None
    mock_db.driver.assert_not_called()

def test_close(mock_neo4j, clean_env):
    _, mock_driver = mock_neo4j
    kg = KnowledgeGraph()
    assert kg.driver is not None
    kg.close()
    mock_driver.close.assert_called_once()
    assert kg.driver is None

def test_close_without_driver(clean_env):
    kg = KnowledgeGraph()
    kg.driver = None
    kg.close() # Should not raise

def test_clear_all(mock_neo4j, clean_env):
    _, mock_driver = mock_neo4j
    mock_session = mock_driver.session.return_value.__enter__.return_value
    mock_result = MagicMock()
    mock_result.consume.return_value.counters.nodes_deleted = 42
    mock_session.run.return_value = mock_result
    
    kg = KnowledgeGraph()
    deleted = kg.clear_all()
    
    assert deleted == 42
    mock_session.run.assert_called_with("MATCH (n) DETACH DELETE n")

def test_clear_all_no_driver():
    kg = KnowledgeGraph()
    kg.driver = None
    assert kg.clear_all() == 0

def test_ensure_layer_root_no_driver():
    kg = KnowledgeGraph()
    kg.driver = None
    kg.ensure_layer_root("NARRATIVE") # Should return silently

def test_add_node(mock_neo4j, clean_env):
    _, mock_driver = mock_neo4j
    mock_session = mock_driver.session.return_value.__enter__.return_value
    
    kg = KnowledgeGraph()
    kg.add_node("Person", "Alice", layer=GraphLayer.SOCIAL, properties={"age": 30})
    
    mock_session.run.assert_called()
    called_args, called_kwargs = mock_session.run.call_args
    assert "MERGE (n:Person" in called_args[0]
    assert "n.age = $age" in called_args[0]
    assert called_kwargs["name"] == "Alice"
    assert called_kwargs["layer"] == GraphLayer.SOCIAL
    assert called_kwargs["age"] == 30

def test_add_node_no_driver():
    kg = KnowledgeGraph()
    kg.driver = None
    kg.add_node("Person", "Alice") # Should return silently

def test_add_relationship(mock_neo4j, clean_env):
    _, mock_driver = mock_neo4j
    mock_session = mock_driver.session.return_value.__enter__.return_value
    
    kg = KnowledgeGraph()
    kg.add_relationship("Alice", "KNOWS", "Bob", strength=0.8)
    
    mock_session.run.assert_called()
    called_args, called_kwargs = mock_session.run.call_args
    assert "MERGE (a)-[r:KNOWS]->(b)" in called_args[0]
    assert called_kwargs["source"] == "Alice"
    assert called_kwargs["target"] == "Bob"
    assert called_kwargs["strength"] == 0.8

def test_add_relationship_sanitize(mock_neo4j, clean_env):
    _, mock_driver = mock_neo4j
    mock_session = mock_driver.session.return_value.__enter__.return_value
    
    kg = KnowledgeGraph()
    kg.add_relationship("Alice", "LIKES!@#", "Bob")
    
    called_args, _ = mock_session.run.call_args
    # Invalid chars removed, converted to uppercase
    assert "MERGE (a)-[r:LIKES]->(b)" in called_args[0]

def test_add_relationship_no_driver():
    kg = KnowledgeGraph()
    kg.driver = None
    kg.add_relationship("A", "B", "C") # Should return silently

def test_add_relationship_empty_sanitize(mock_neo4j, clean_env):
    _, mock_driver = mock_neo4j
    mock_session = mock_driver.session.return_value.__enter__.return_value
    
    kg = KnowledgeGraph()
    kg.add_relationship("A", "!@#", "B")
    
    called_args, _ = mock_session.run.call_args
    assert "MERGE (a)-[r:RELATED_TO]->(b)" in called_args[0]

def test_query_context(mock_neo4j, clean_env):
    _, mock_driver = mock_neo4j
    mock_session = mock_driver.session.return_value.__enter__.return_value
    mock_record = {"a_name": "Alice", "a_layer": GraphLayer.NARRATIVE, "rel": "KNOWS", "strength": 1.0, "b_name": "Bob", "b_layer": GraphLayer.NARRATIVE, "r_user_id": "", "r_scope": "PUBLIC"}
    mock_session.run.return_value = [mock_record]
    
    kg = KnowledgeGraph()
    results = kg.query_context("Alice")
    
    assert len(results) == 1
    assert results[0]["source"] == "Alice"
    assert results[0]["target"] == "Bob"
    
def test_query_context_with_layer(mock_neo4j, clean_env):
    _, mock_driver = mock_neo4j
    mock_session = mock_driver.session.return_value.__enter__.return_value
    mock_session.run.return_value = []
    
    kg = KnowledgeGraph()
    kg.query_context("Alice", layer=GraphLayer.SYSTEM)
    
    called_args, called_kwargs = mock_session.run.call_args
    assert "AND a.layer = $layer" in called_args[0]
    assert called_kwargs["layer"] == GraphLayer.SYSTEM

def test_query_context_no_driver():
    kg = KnowledgeGraph()
    kg.driver = None
    assert kg.query_context("Alice") == []

def test_initialize_schema(mock_neo4j, clean_env):
    _, mock_driver = mock_neo4j
    mock_session = mock_driver.session.return_value.__enter__.return_value
    
    kg = KnowledgeGraph()
    # It gets called in connect()
    assert mock_session.run.call_count >= 2 # Should run multiple schema building queries

def test_initialize_schema_no_driver(clean_env):
    kg = KnowledgeGraph()
    kg.driver = None
    kg._initialize_schema() # Should return silently
