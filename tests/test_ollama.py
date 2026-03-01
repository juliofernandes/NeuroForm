import pytest
from unittest.mock import MagicMock, patch
from neuroform.memory.graph import KnowledgeGraph, GraphLayer
from neuroform.llm.ollama_client import OllamaClient

@pytest.fixture
def mock_kg():
    kg = MagicMock(spec=KnowledgeGraph)
    kg.driver = MagicMock()
    return kg

@patch("neuroform.llm.ollama_client.ollama.chat")
def test_chat_with_memory_no_context_no_json(mock_chat, mock_kg):
    mock_kg.query_context.return_value = []
    mock_chat.return_value = {"message": {"content": "Hello! I am fine."}}
    
    client = OllamaClient(mock_kg, model="test_model")
    reply = client.chat_with_memory("user123", "How are you?")
    
    assert reply == "Hello! I am fine."
    mock_kg.query_context.assert_called_once_with("User", layer=GraphLayer.NARRATIVE)
    
    # Assert ollama was called
    mock_chat.assert_called_once()
    args, kwargs = mock_chat.call_args
    assert kwargs["model"] == "test_model"
    # The system prompt should contain the user's message from working memory
    # (WorkingMemory always has at least the user's conversation turn)
    assert "Prior Memory:" in kwargs["messages"][0]["content"]

@patch("neuroform.llm.ollama_client.ollama.chat")
def test_chat_with_memory_with_context(mock_chat, mock_kg):
    mock_kg.query_context.return_value = [
        {"source": "User", "relationship": "LIKES", "target": "Pizza", "strength": 1.0}
    ]
    mock_chat.return_value = {"message": {"content": "I remember you like Pizza!"}}
    
    client = OllamaClient(mock_kg)
    reply = client.chat_with_memory("user123", "What do I like?")
    
    assert reply == "I remember you like Pizza!"
    args, kwargs = mock_chat.call_args
    # WorkingMemory formats graph context as "[GRAPH] source (relation) target"
    assert "User (LIKES) Pizza" in kwargs["messages"][0]["content"]

@patch("neuroform.llm.ollama_client.ollama.chat")
def test_chat_with_memory_with_json_extraction(mock_chat, mock_kg):
    mock_kg.query_context.return_value = []
    
    ai_reply = '''I note that!
```json
{
    "new_memories": [
        {"source": "User", "relation": "HAS_DOG", "target": "Buddy", "layer": "SOCIAL"}
    ]
}
```'''
    mock_chat.return_value = {"message": {"content": ai_reply}}
    
    client = OllamaClient(mock_kg)
    reply = client.chat_with_memory("user123", "I have a dog named Buddy.")
    
    assert reply == "I note that!"
    
    # Verify nodes and relationship were created
    assert mock_kg.add_node.call_count == 2
    mock_kg.add_relationship.assert_called_once_with("User", "HAS_DOG", "Buddy", strength=1.0)
    
@patch("neuroform.llm.ollama_client.ollama.chat")
def test_chat_with_memory_bad_json(mock_chat, mock_kg):
    mock_kg.query_context.return_value = []
    
    ai_reply = '''Sure!
```json
{
    "new_memories": [ Broken JSON
```'''
    mock_chat.return_value = {"message": {"content": ai_reply}}
    
    client = OllamaClient(mock_kg)
    reply = client.chat_with_memory("user123", "I like cars.")
    
    assert reply == "Sure!"
    mock_kg.add_relationship.assert_not_called()

@patch("neuroform.llm.ollama_client.ollama.chat")
def test_chat_with_memory_exception(mock_chat, mock_kg):
    mock_chat.side_effect = Exception("Ollama is down")
    
    client = OllamaClient(mock_kg)
    reply = client.chat_with_memory("user123", "Hello")
    
    assert reply == "I am having trouble connecting to my neural core right now."

def test_extract_and_save_memories_exception(mock_kg):
    client = OllamaClient(mock_kg)
    # Give mock_kg an exception mechanism
    mock_kg.add_node.side_effect = Exception("DB error")
    
    ai_reply = '''```json
{"new_memories": [{"source": "U", "relation": "R", "target": "T"}]}
```'''
    
    # Needs to not raise uncaught exception
    client._extract_and_save_memories(ai_reply)
    mock_kg.add_node.assert_called_once()

@patch("neuroform.llm.ollama_client.ollama.chat")
def test_chat_records_conversation_turns(mock_chat, mock_kg):
    """Verify that chat adds both user and assistant turns to working memory."""
    mock_kg.query_context.return_value = []
    mock_chat.return_value = {"message": {"content": "Hi there!"}}
    
    client = OllamaClient(mock_kg)
    client.chat_with_memory("user123", "Hello")
    
    history = client.working_memory.get_conversation_history()
    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "Hello"}
    assert history[1] == {"role": "assistant", "content": "Hi there!"}

@patch("neuroform.llm.ollama_client.ollama.chat")
def test_chat_multi_turn_history_in_prompt(mock_chat, mock_kg):
    """Verify that multi-turn conversation history is sent to the LLM."""
    mock_kg.query_context.return_value = []
    mock_chat.return_value = {"message": {"content": "Response 1"}}
    
    client = OllamaClient(mock_kg)
    client.chat_with_memory("user123", "Turn 1")
    
    mock_chat.return_value = {"message": {"content": "Response 2"}}
    client.chat_with_memory("user123", "Turn 2")
    
    # The second call should include history from turn 1
    args, kwargs = mock_chat.call_args
    messages = kwargs["messages"]
    # Should be: system, user("Turn 1"), assistant("Response 1"), user("Turn 2")
    assert len(messages) >= 3

@patch("neuroform.llm.ollama_client.ollama.chat")
def test_custom_working_memory_injected(mock_chat, mock_kg):
    """Verify that a custom WorkingMemory instance is used when provided."""
    from neuroform.memory.working_memory import WorkingMemory
    custom_wm = WorkingMemory(capacity=3)
    
    mock_kg.query_context.return_value = []
    mock_chat.return_value = {"message": {"content": "ok"}}
    
    client = OllamaClient(mock_kg, working_memory=custom_wm)
    assert client.working_memory is custom_wm
    assert client.working_memory.capacity == 3
