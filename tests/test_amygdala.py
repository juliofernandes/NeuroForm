"""
Unit Tests for Amygdala — Emotional Valence Tagging
====================================================
100% coverage of EmotionCategory, EmotionalValence, and Amygdala classes.
"""
import pytest
from unittest.mock import MagicMock, patch
from neuroform.memory.amygdala import Amygdala, EmotionalValence, EmotionCategory


# ===========================================================================
# EmotionCategory Tests
# ===========================================================================
class TestEmotionCategory:

    def test_all_categories_are_strings(self):
        for cat in EmotionCategory:
            assert isinstance(cat.value, str)

    def test_neutral_exists(self):
        assert EmotionCategory.NEUTRAL == "neutral"

    def test_joy_exists(self):
        assert EmotionCategory.JOY == "joy"


# ===========================================================================
# EmotionalValence Tests
# ===========================================================================
class TestEmotionalValence:

    def test_defaults(self):
        v = EmotionalValence()
        assert v.valence == 0.0
        assert v.intensity == 0.0
        assert v.emotion == "neutral"

    def test_clamping_high(self):
        v = EmotionalValence(valence=5.0, intensity=3.0)
        assert v.valence == 1.0
        assert v.intensity == 1.0

    def test_clamping_low(self):
        v = EmotionalValence(valence=-5.0, intensity=-1.0)
        assert v.valence == -1.0
        assert v.intensity == 0.0

    def test_is_significant_high_valence(self):
        v = EmotionalValence(valence=0.8, intensity=0.3)
        assert v.is_significant is True

    def test_is_significant_high_intensity(self):
        v = EmotionalValence(valence=0.3, intensity=0.8)
        assert v.is_significant is True

    def test_is_significant_negative_valence(self):
        v = EmotionalValence(valence=-0.9, intensity=0.5)
        assert v.is_significant is True

    def test_not_significant(self):
        v = EmotionalValence(valence=0.3, intensity=0.3)
        assert v.is_significant is False

    def test_survival_bonus_significant(self):
        v = EmotionalValence(valence=0.9, intensity=0.8)
        bonus = v.survival_bonus
        assert bonus > 0
        assert bonus == pytest.approx(0.9 * 0.8 * 0.5, abs=0.01)

    def test_survival_bonus_not_significant(self):
        v = EmotionalValence(valence=0.2, intensity=0.2)
        assert v.survival_bonus == 0.0

    def test_to_dict(self):
        v = EmotionalValence(valence=0.8, intensity=0.6, emotion="joy")
        d = v.to_dict()
        assert d["valence"] == 0.8
        assert d["intensity"] == 0.6
        assert d["emotion"] == "joy"
        assert d["is_significant"] is True
        assert "survival_bonus" in d

    def test_from_dict(self):
        d = {"valence": -0.5, "intensity": 0.3, "emotion": "sadness"}
        v = EmotionalValence.from_dict(d)
        assert v.valence == -0.5
        assert v.intensity == 0.3
        assert v.emotion == "sadness"

    def test_from_dict_defaults(self):
        v = EmotionalValence.from_dict({})
        assert v.valence == 0.0
        assert v.intensity == 0.0
        assert v.emotion == "neutral"

    def test_repr(self):
        v = EmotionalValence(valence=0.5, intensity=0.3, emotion="joy")
        r = repr(v)
        assert "0.50" in r
        assert "0.30" in r
        assert "joy" in r


# ===========================================================================
# Amygdala Tests
# ===========================================================================
class TestAmygdala:

    def test_init_defaults(self):
        a = Amygdala()
        assert a.decay_immunity_threshold == 0.7

    def test_init_custom_threshold(self):
        a = Amygdala(decay_immunity_threshold=0.5)
        assert a.decay_immunity_threshold == 0.5

    def test_extract_valence_with_data(self):
        a = Amygdala()
        mem = {"valence": 0.8, "intensity": 0.6, "emotion": "joy"}
        v = a.extract_valence(mem)
        assert v.valence == 0.8
        assert v.intensity == 0.6
        assert v.emotion == "joy"

    def test_extract_valence_missing_fields(self):
        a = Amygdala()
        v = a.extract_valence({})
        assert v.valence == 0.0
        assert v.intensity == 0.0
        assert v.emotion == "neutral"

    def test_should_protect_significant(self):
        a = Amygdala()
        v = EmotionalValence(valence=0.9, intensity=0.8)
        assert a.should_protect_from_decay(v) is True

    def test_should_not_protect_insignificant(self):
        a = Amygdala()
        v = EmotionalValence(valence=0.1, intensity=0.1)
        assert a.should_protect_from_decay(v) is False

    def test_apply_valence_to_edge(self):
        a = Amygdala()
        mock_session = MagicMock()
        v = EmotionalValence(valence=-0.8, intensity=0.9, emotion="fear")
        a.apply_valence_to_edge(mock_session, "User", "EXPERIENCED", "Trauma", v)
        mock_session.run.assert_called_once()
        call_kwargs = mock_session.run.call_args
        assert call_kwargs[1]["valence"] == -0.8
        assert call_kwargs[1]["intensity"] == 0.9
        assert call_kwargs[1]["emotion"] == "fear"

    def test_apply_valence_sanitizes_rel_type(self):
        a = Amygdala()
        mock_session = MagicMock()
        v = EmotionalValence()
        a.apply_valence_to_edge(mock_session, "A", "has been!!!", "B", v)
        call_args = mock_session.run.call_args[0][0]
        assert "HASBEEN" in call_args

    def test_apply_valence_empty_rel_type(self):
        a = Amygdala()
        mock_session = MagicMock()
        v = EmotionalValence()
        a.apply_valence_to_edge(mock_session, "A", "!!!", "B", v)
        call_args = mock_session.run.call_args[0][0]
        assert "RELATED_TO" in call_args

    def test_get_decay_immunity_cypher(self):
        a = Amygdala(decay_immunity_threshold=0.7)
        clause = a.get_decay_immunity_cypher()
        assert "0.7" in clause
        assert "valence" in clause
        assert "intensity" in clause
        assert clause.startswith("AND")

    def test_tag_memories_no_driver(self):
        a = Amygdala()
        # Should not crash
        a.tag_memories(None, [{"source": "A", "relation": "R", "target": "B", "valence": 0.5}])

    def test_tag_memories_with_emotion(self):
        a = Amygdala()
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        memories = [
            {"source": "User", "relation": "LOST", "target": "Dog",
             "valence": -0.9, "intensity": 0.8, "emotion": "sadness"}
        ]
        a.tag_memories(mock_driver, memories)
        mock_session.run.assert_called_once()

    def test_tag_memories_skips_neutral(self):
        a = Amygdala()
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        memories = [
            {"source": "User", "relation": "SAW", "target": "Tree",
             "valence": 0.0, "emotion": "neutral"}
        ]
        a.tag_memories(mock_driver, memories)
        mock_session.run.assert_not_called()

    def test_tag_memories_skips_incomplete(self):
        a = Amygdala()
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        memories = [{"source": "User", "valence": 0.8}]  # Missing relation and target
        a.tag_memories(mock_driver, memories)
        mock_session.run.assert_not_called()

    def test_tag_memories_handles_exception(self):
        a = Amygdala()
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_session.run.side_effect = Exception("DB error")
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        memories = [
            {"source": "User", "relation": "LOST", "target": "Dog",
             "valence": -0.9, "intensity": 0.8, "emotion": "sadness"}
        ]
        # Should not raise
        a.tag_memories(mock_driver, memories)

    def test_valence_extraction_prompt_exists(self):
        """Verify the prompt fragment is well-formed."""
        assert "valence" in Amygdala.VALENCE_EXTRACTION_PROMPT
        assert "intensity" in Amygdala.VALENCE_EXTRACTION_PROMPT
        assert "emotion" in Amygdala.VALENCE_EXTRACTION_PROMPT
