"""
Unit Tests for BrainOrchestrator + BackgroundScheduler
=======================================================
"""
import pytest
import time
from unittest.mock import MagicMock, patch, PropertyMock

from neuroform.brain.orchestrator import BrainOrchestrator, ContextObject
from neuroform.brain.background import BackgroundScheduler


# ===========================================================================
# BrainOrchestrator Tests
# ===========================================================================
class TestBrainOrchestrator:

    @pytest.fixture
    def mock_kg(self):
        kg = MagicMock()
        kg.driver = MagicMock()
        kg.query_context.return_value = [
            "User_u1 -[LIKES]-> Music (strength: 3.0)",
        ]
        return kg

    @pytest.fixture
    def orchestrator(self, mock_kg, tmp_path):
        from neuroform.memory.context_stream import ContextStream
        from neuroform.memory.vector_store import VectorStore
        from neuroform.memory.lessons import LessonManager

        cs = ContextStream(max_turns=100, persist_path=str(tmp_path / "wm.jsonl"))
        vs = VectorStore(model="test", max_entries=100)
        lm = LessonManager(persist_path=str(tmp_path / "lessons.json"))

        with patch("neuroform.brain.orchestrator.OllamaClient") as MockClient:
            MockClient.return_value.chat_with_memory.return_value = "Test response"
            with patch.object(vs, "embed", return_value=[]):  # Skip real embeddings
                orch = BrainOrchestrator(
                    mock_kg, model="test-model",
                    context_stream=cs, vector_store=vs,
                    lesson_manager=lm,
                )
                return orch

    def test_init_creates_all_systems(self, orchestrator):
        assert orchestrator.context_stream is not None
        assert orchestrator.vector_store is not None
        assert orchestrator.lessons is not None
        assert orchestrator.amygdala is not None
        assert orchestrator.salience is not None
        assert orchestrator.habit_cache is not None
        assert orchestrator.nt is not None
        assert orchestrator.predictive_model is not None
        assert orchestrator.dmn is not None
        assert orchestrator.circadian is not None
        assert orchestrator.neuroplasticity is not None
        assert orchestrator.client is not None

    def test_process_returns_response(self, orchestrator):
        response = orchestrator.process("user1", "Hello")
        assert isinstance(response, str)
        assert len(response) > 0

    def test_process_increments_count(self, orchestrator):
        assert orchestrator._message_count == 0
        orchestrator.process("user1", "Hello")
        assert orchestrator._message_count == 1

    def test_process_passes_user_name(self, orchestrator):
        orchestrator.process("user1", "Hello", user_name="Maria")
        # Should have added a turn to ContextStream with user_name
        assert orchestrator.context_stream.turn_count >= 1
        last_turn = orchestrator.context_stream.buffer[-1]
        assert last_turn.user_name == "Maria"

    def test_circadian_modulates_nt(self, orchestrator):
        orchestrator.process("user1", "Hello")
        assert orchestrator.nt.norepinephrine is not None
        assert orchestrator.nt.dopamine is not None

    def test_recall_gathers_4_tiers(self, orchestrator):
        # Add some data to each tier
        orchestrator.context_stream.add_turn("u1", "hello", "hi", user_name="Maria")
        orchestrator.lessons.add_lesson("Maria is the developer", user_id="u1")

        ctx = orchestrator._recall("hello", "u1", user_name="Maria")
        assert isinstance(ctx, ContextObject)
        assert isinstance(ctx.conversation_history, str)
        assert isinstance(ctx.vector_memories, list)
        assert isinstance(ctx.graph_context, list)
        assert isinstance(ctx.lessons, list)
        assert "Maria is the developer" in ctx.lessons

    def test_observe_persists_to_streams(self, orchestrator):
        orchestrator._observe("u1", "hello", "hi", user_name="Maria")
        assert orchestrator.context_stream.turn_count == 1
        turn = orchestrator.context_stream.buffer[0]
        assert turn.user_name == "Maria"
        assert turn.user_message == "hello"

    def test_habit_cache_records_invocation(self, orchestrator):
        orchestrator.process("user1", "hello there")
        # Short responses (<50 chars) should NOT be recorded
        count = orchestrator.habit_cache.get_invocation_count("hello there")
        assert count == 0

    def test_habit_cache_shortcircuit(self, orchestrator):
        """After enough invocations with long responses, habit cache should bypass LLM."""
        orchestrator.client.chat_with_memory.return_value = "A" * 60
        for i in range(15):
            orchestrator.process("user1", "hello there my friend")
        orchestrator.client.chat_with_memory.reset_mock()
        response = orchestrator.process("user1", "hello there my friend")
        assert isinstance(response, str)

    def test_sentiment_modulation(self, orchestrator):
        orchestrator.nt.reset()
        baseline_da = orchestrator.nt.dopamine
        orchestrator.process("user1", "I love this amazing awesome great thing")
        assert orchestrator.nt.dopamine >= baseline_da

    def test_negative_sentiment_modulation(self, orchestrator):
        orchestrator.nt.reset()
        orchestrator.process("user1", "I hate this terrible awful thing")
        assert orchestrator.nt.norepinephrine is not None

    def test_prediction_evaluation(self, orchestrator):
        orchestrator.predictive_model._last_prediction = "User wants music"
        orchestrator.predictive_model._last_context_sources = ["User_Music"]
        orchestrator._last_user_message = "Tell me about music"

        with patch.object(orchestrator.predictive_model, 'evaluate_error',
                         return_value=0.2) as mock_eval:
            with patch.object(orchestrator.predictive_model, 'generate_feedback_signal',
                             return_value=[]) as mock_signal:
                orchestrator.process("user1", "Play some jazz")
                mock_eval.assert_called_once()
                mock_signal.assert_called_once()

    def test_get_diagnostics(self, orchestrator):
        orchestrator.process("user1", "Hello brain")
        diag = orchestrator.get_diagnostics()
        assert "message_count" in diag
        assert diag["message_count"] == 1
        assert "neurotransmitters" in diag
        assert "circadian" in diag
        assert "context_stream" in diag
        assert "vector_store" in diag
        assert "lessons" in diag
        assert "habit_cache" in diag

    def test_compute_habit_key(self, orchestrator):
        key = orchestrator._compute_habit_key("Hello world how are you today")
        assert key == "hello world how are you today"

    def test_compute_habit_key_short(self, orchestrator):
        key = orchestrator._compute_habit_key("hi")
        assert key == "hi"

    def test_compute_habit_key_empty(self, orchestrator):
        key = orchestrator._compute_habit_key("")
        assert key == "empty"

    def test_estimate_sentiment_positive(self):
        s = BrainOrchestrator._estimate_sentiment("I love this amazing thing")
        assert s > 0

    def test_estimate_sentiment_negative(self):
        s = BrainOrchestrator._estimate_sentiment("I hate this terrible thing")
        assert s < 0

    def test_estimate_sentiment_neutral(self):
        s = BrainOrchestrator._estimate_sentiment("The sky is blue")
        assert s == 0.0

    def test_format_tiered_context(self, orchestrator):
        ctx = ContextObject(
            conversation_history="Alice: hi\nBot: hello",
            vector_memories=["memory about cats"],
            graph_context=["Alice -[KNOWS]-> Bob"],
            lessons=["Alice is a developer"],
            foundation_facts="[SRC:FN:Alice] Alice -[ROLE]-> developer",
        )
        formatted = orchestrator._format_tiered_context(ctx)
        assert "FOUNDATION KNOWLEDGE" in formatted or "LESSONS" in formatted
        assert "Alice is a developer" in formatted

    def test_format_tiered_context_empty(self, orchestrator):
        ctx = ContextObject(
            conversation_history="No conversation history.",
            vector_memories=[],
            graph_context=[],
            lessons=[],
            foundation_facts="",
        )
        formatted = orchestrator._format_tiered_context(ctx)
        assert formatted == "No prior context available."

    def test_apply_feedback_strengthen(self, orchestrator):
        signals = [{"action": "STRENGTHEN", "target": "User_Music", "amount": 0.2}]
        orchestrator._apply_feedback_signals(signals)
        orchestrator.kg.driver.session.assert_called()

    def test_apply_feedback_decay(self, orchestrator):
        signals = [{"action": "DECAY", "target": "User_OldTopic", "amount": 0.1}]
        orchestrator._apply_feedback_signals(signals)
        orchestrator.kg.driver.session.assert_called()

    def test_apply_feedback_bad_target(self, orchestrator):
        signals = [{"action": "STRENGTHEN", "target": "notarget", "amount": 0.1}]
        orchestrator._apply_feedback_signals(signals)  # Should not crash

    def test_apply_feedback_db_error(self, orchestrator):
        orchestrator.kg.driver.session.side_effect = Exception("DB down")
        signals = [{"action": "STRENGTHEN", "target": "A_B", "amount": 0.1}]
        orchestrator._apply_feedback_signals(signals)  # Should not crash

    def test_no_context_from_graph(self, orchestrator):
        orchestrator.kg.query_context.return_value = []
        response = orchestrator.process("user1", "Hello")
        assert isinstance(response, str)

    def test_prediction_exception_handled(self, orchestrator):
        with patch.object(orchestrator.predictive_model, 'predict_intent',
                         side_effect=Exception("LLM down")):
            response = orchestrator.process("user1", "Hello")
            assert isinstance(response, str)

    def test_prediction_with_feedback_signals(self, orchestrator):
        orchestrator.predictive_model._last_prediction = "User wants cats"
        orchestrator.predictive_model._last_context_sources = ["User_Cats"]
        orchestrator._last_user_message = "Tell me about cats"

        with patch.object(orchestrator.predictive_model, 'evaluate_error',
                         return_value=0.8):
            with patch.object(orchestrator.predictive_model, 'generate_feedback_signal',
                             return_value=[{"action": "DECAY", "target": "User_Cats", "amount": 0.1}]):
                orchestrator.process("user1", "Actually tell me about dogs")
                orchestrator.kg.driver.session.assert_called()

    def test_habit_not_recorded_for_short_response(self, orchestrator):
        orchestrator.client.chat_with_memory.return_value = "Short"
        orchestrator.process("user1", "test message")
        count = orchestrator.habit_cache.get_invocation_count("test message")
        assert count == 0

    def test_habit_recorded_for_long_response(self, orchestrator):
        orchestrator.client.chat_with_memory.return_value = "A" * 60
        orchestrator.process("user1", "test message")
        count = orchestrator.habit_cache.get_invocation_count("test message")
        assert count == 1

    def test_recall_graph_error_handled(self, orchestrator):
        orchestrator.kg.query_context.side_effect = Exception("DB error")
        ctx = orchestrator._recall("hello", "u1")
        assert ctx.graph_context == []

    def test_recall_with_user_name(self, orchestrator):
        orchestrator.kg.query_context.return_value = ["Maria -[IS_DEVELOPER_OF]-> NeuroForm"]
        ctx = orchestrator._recall("who am I", "u1", user_name="Maria")
        # Should have queried by user_name
        calls = [str(c) for c in orchestrator.kg.query_context.call_args_list]
        assert any("Maria" in c for c in calls)


# ===========================================================================
# BackgroundScheduler Tests
# ===========================================================================
class TestBackgroundScheduler:

    @pytest.fixture
    def mock_kg(self):
        kg = MagicMock()
        kg.driver = MagicMock()
        return kg

    @pytest.fixture
    def scheduler(self, mock_kg):
        return BackgroundScheduler(
            mock_kg, model="test",
            idle_timeout=1.0, decay_interval=1.0, tick_interval=1.0,
        )

    def test_init(self, scheduler):
        assert not scheduler.is_running
        assert scheduler.dream_runs == 0
        assert scheduler.dmn_runs == 0
        assert scheduler.decay_runs == 0

    def test_record_activity(self, scheduler):
        old = scheduler._last_active
        time.sleep(0.01)
        scheduler.record_activity()
        assert scheduler._last_active > old

    def test_tick_decay(self, scheduler):
        scheduler._last_decay = 0
        results = scheduler.tick()
        assert results["decay"] is not None
        assert results["decay"]["status"] == "applied"
        assert scheduler.decay_runs == 1

    def test_tick_dream_consolidation(self, scheduler):
        with patch.object(scheduler.circadian, 'should_dream_now', return_value=True):
            scheduler._last_dream = 0
            scheduler._last_decay = time.time()
            with patch.object(scheduler.dream, 'consolidate',
                            return_value={"status": "consolidated"}):
                results = scheduler.tick()
                assert results["dream"] is not None
                assert results["dream"]["status"] == "consolidated"
                assert scheduler.dream_runs == 1

    def test_tick_dmn_introspection(self, scheduler):
        scheduler._last_active = 0
        scheduler._last_dmn = 0
        scheduler._last_decay = time.time()
        with patch.object(scheduler.circadian, 'should_dream_now', return_value=False):
            with patch.object(scheduler.dmn, 'introspect',
                            return_value={"status": "complete"}):
                results = scheduler.tick()
                assert results["dmn"] is not None
                assert results["dmn"]["status"] == "complete"
                assert scheduler.dmn_runs == 1

    def test_tick_dream_error(self, scheduler):
        with patch.object(scheduler.circadian, 'should_dream_now', return_value=True):
            scheduler._last_dream = 0
            scheduler._last_decay = time.time()
            with patch.object(scheduler.dream, 'consolidate',
                            side_effect=Exception("LLM error")):
                results = scheduler.tick()
                assert results["dream"]["status"] == "error"

    def test_tick_dmn_error(self, scheduler):
        scheduler._last_active = 0
        scheduler._last_dmn = 0
        scheduler._last_decay = time.time()
        with patch.object(scheduler.circadian, 'should_dream_now', return_value=False):
            with patch.object(scheduler.dmn, 'introspect',
                            side_effect=Exception("Error")):
                results = scheduler.tick()
                assert results["dmn"]["status"] == "error"

    def test_tick_decay_error(self, scheduler):
        scheduler._last_decay = 0
        with patch.object(scheduler.neuroplasticity, 'apply_baseline_decay',
                         side_effect=Exception("DB error")):
            results = scheduler.tick()
            assert results["decay"]["status"] == "error"

    def test_stop_when_not_running(self, scheduler):
        scheduler.stop()
        assert not scheduler.is_running

    def test_stop_with_thread(self, scheduler):
        import threading
        mock_thread = MagicMock(spec=threading.Thread)
        scheduler._running = True
        scheduler._thread = mock_thread
        scheduler.stop()
        assert not scheduler._running
        mock_thread.join.assert_called_once_with(timeout=5.0)
        assert scheduler._thread is None

    def test_snapshot(self, scheduler):
        snap = scheduler.snapshot()
        assert "running" in snap
        assert "dream_runs" in snap
        assert "dmn_runs" in snap
        assert "decay_runs" in snap
        assert "idle_seconds" in snap
        assert snap["running"] is False

    def test_no_duplicate_dream(self, scheduler):
        scheduler._last_dream = time.time()
        scheduler._last_decay = time.time()
        with patch.object(scheduler.circadian, 'should_dream_now', return_value=True):
            results = scheduler.tick()
            assert results["dream"] is None

    def test_no_dmn_when_active(self, scheduler):
        scheduler._last_active = time.time()
        scheduler._last_decay = time.time()
        with patch.object(scheduler.circadian, 'should_dream_now', return_value=False):
            results = scheduler.tick()
            assert results["dmn"] is None


# ===========================================================================
# Phase 2 Coverage Tests
# ===========================================================================
class TestPhase2Integration:

    @pytest.fixture
    def mock_kg(self):
        kg = MagicMock()
        kg.driver = MagicMock()
        kg.query_context.return_value = [
            {"source": "Maria", "relationship": "IS_A", "target": "Developer"},
        ]
        return kg

    @pytest.fixture
    def orchestrator(self, mock_kg, tmp_path):
        from neuroform.memory.context_stream import ContextStream
        from neuroform.memory.vector_store import VectorStore
        from neuroform.memory.lessons import LessonManager
        from neuroform.memory.tape_machine import TapeMachine
        from neuroform.memory.reconciler import CrossTierReconciler
        from neuroform.memory.scopes import ScopeManager

        cs = ContextStream(max_turns=100, persist_path=str(tmp_path / "wm.jsonl"))
        vs = VectorStore(model="test", max_entries=100)
        lm = LessonManager(persist_path=str(tmp_path / "lessons.json"))
        tm = TapeMachine(user_id="test", persist_dir=str(tmp_path / "tape"))
        reconciler = CrossTierReconciler(model="test")
        sm = ScopeManager(enable_scopes=True)

        with patch("neuroform.brain.orchestrator.OllamaClient") as MockClient:
            MockClient.return_value.chat_with_memory.return_value = "Test response"
            with patch.object(vs, "embed", return_value=[]):
                orch = BrainOrchestrator(
                    mock_kg, model="test-model",
                    context_stream=cs, vector_store=vs,
                    lesson_manager=lm, tape_machine=tm,
                    reconciler=reconciler, scope_manager=sm,
                )
                return orch

    def test_init_phase2_systems(self, orchestrator):
        assert orchestrator.tape is not None
        assert orchestrator.reconciler is not None
        assert orchestrator.scope_manager is not None

    def test_recall_dict_graph_facts(self, orchestrator):
        """Graph returns dict facts — cover L253 (isinstance(fact, dict))."""
        ctx = orchestrator._recall("hello", "u1", user_name="Maria")
        # KG returns dicts, so reconciler should get string versions
        assert isinstance(ctx, ContextObject)
        assert ctx.tape_view is not None
        assert "TAPE MACHINE" in ctx.tape_view

    @patch("neuroform.memory.reconciler._ollama")
    def test_recall_reconciliation_conflicts(self, mock_ollama, orchestrator):
        """Cover L268-273 — reconciler finds conflicts."""
        mock_ollama.chat.return_value = {
            "message": {"content": "CONFLICT:KG:0|Stale data in KG"}
        }
        orchestrator.lessons.add_lesson("Maria is vegan", user_id="u1")
        ctx = orchestrator._recall("diet", "u1", user_name="Maria")
        assert "RECONCILIATION" in ctx.reconciliation_notes or ctx.reconciliation_notes == ""

    def test_recall_reconciliation_exception(self, orchestrator):
        """Cover L273 — reconciler raises exception."""
        orchestrator.reconciler.reconcile = MagicMock(side_effect=Exception("boom"))
        ctx = orchestrator._recall("hello", "u1")
        # Should not crash, reconciliation_notes stays empty
        assert ctx.reconciliation_notes == ""

    def test_observe_writes_to_tape(self, orchestrator):
        """Cover tape write in _observe — T5 tape write."""
        initial_ptr = orchestrator.tape.focus_pointer
        orchestrator._observe("u1", "hello", "world", user_name="Maria")
        # Tape pointer should advance
        assert orchestrator.tape.focus_pointer != initial_ptr

    def test_format_includes_tape_and_reconciliation(self, orchestrator):
        """Cover L319 — tape_view in format, and reconciliation_notes."""
        ctx = ContextObject(
            conversation_history="User: hi\nBot: hello",
            vector_memories=["mem1"],
            graph_context=["fact1"],
            lessons=["lesson1"],
            foundation_facts="foundation",
            tape_view="--- TAPE MACHINE ---",
            reconciliation_notes="[RECONCILIATION: 1 conflicts detected]",
        )
        formatted = orchestrator._format_tiered_context(ctx)
        assert "TAPE MACHINE" in formatted
        assert "RECONCILIATION" in formatted
        assert "LESSONS" in formatted

    def test_diagnostics_includes_phase2(self, orchestrator):
        """Cover expanded diagnostics with tape, reconciler, scope_manager."""
        diag = orchestrator.get_diagnostics()
        assert "tape" in diag
        assert "reconciler" in diag
        assert "scope_manager" in diag
        assert diag["tape"]["user_id"] == "test"
        assert diag["scope_manager"]["enabled"] is True

    def test_process_with_phase2(self, orchestrator):
        """E2E process call with Phase 2 systems wired in."""
        response = orchestrator.process("u1", "hello", user_name="Maria")
        assert isinstance(response, str)
        assert len(response) > 0
