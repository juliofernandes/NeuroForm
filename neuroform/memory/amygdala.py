"""
Amygdala — Emotional Valence Tagging
====================================

Biological basis: The amygdala tags incoming stimuli with emotional significance
(fear, reward, disgust, joy). Emotionally tagged memories are consolidated
preferentially — you remember your wedding day, not a random Tuesday.

Computational analogue: A valence scoring system that tags graph relationships
with emotional weight. High-valence memories receive decay immunity during
the baseline heuristic phase, ensuring emotionally significant facts survive
longer. The LLM classifies emotional weight during memory extraction.
"""
import logging
from typing import Dict, Any, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class EmotionCategory(str, Enum):
    """Core emotional categories based on Ekman's basic emotions."""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    TRUST = "trust"
    ANTICIPATION = "anticipation"
    NEUTRAL = "neutral"


class EmotionalValence:
    """
    Represents the emotional weight of a memory.

    Valence ranges from -1.0 (extremely negative) to +1.0 (extremely positive).
    Intensity (arousal) ranges from 0.0 (calm) to 1.0 (highly activated).
    """

    def __init__(self, valence: float = 0.0, intensity: float = 0.0,
                 emotion: str = "neutral"):
        self.valence = max(-1.0, min(1.0, valence))
        self.intensity = max(0.0, min(1.0, intensity))
        self.emotion = emotion

    @property
    def is_significant(self) -> bool:
        """Returns True if the emotional valence makes this memory decay-immune."""
        return abs(self.valence) >= 0.7 or self.intensity >= 0.7

    @property
    def survival_bonus(self) -> float:
        """
        Extra strength bonus applied to high-valence edges during decay.
        Significant memories get a positive bonus that counteracts decay.
        """
        if not self.is_significant:
            return 0.0
        return abs(self.valence) * self.intensity * 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valence": self.valence,
            "intensity": self.intensity,
            "emotion": self.emotion,
            "is_significant": self.is_significant,
            "survival_bonus": self.survival_bonus,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmotionalValence":
        return cls(
            valence=data.get("valence", 0.0),
            intensity=data.get("intensity", 0.0),
            emotion=data.get("emotion", "neutral"),
        )

    def __repr__(self):
        return f"EmotionalValence(v={self.valence:.2f}, i={self.intensity:.2f}, e={self.emotion})"


class Amygdala:
    """
    The emotional tagging system for the knowledge graph.

    Responsibilities:
    1. Parse valence data from LLM extraction output
    2. Apply emotional tags to graph relationships
    3. Shield high-valence edges from baseline decay
    """

    VALENCE_EXTRACTION_PROMPT = """Additionally, for each new memory, also rate its emotional significance:
- "valence": float from -1.0 (very negative) to +1.0 (very positive), 0.0 = neutral
- "intensity": float from 0.0 (trivial) to 1.0 (life-changing)
- "emotion": one of: joy, sadness, anger, fear, surprise, disgust, trust, anticipation, neutral

Example:
```json
{
    "new_memories": [
        {"source": "User", "relation": "LOST", "target": "Dog", "layer": "SOCIAL",
         "valence": -0.9, "intensity": 0.8, "emotion": "sadness"}
    ]
}
```"""

    def __init__(self, decay_immunity_threshold: float = 0.7):
        self.decay_immunity_threshold = decay_immunity_threshold

    def extract_valence(self, memory_dict: Dict[str, Any]) -> EmotionalValence:
        """
        Extract emotional valence data from a parsed memory dict.
        Falls back to neutral if fields are missing.
        """
        return EmotionalValence(
            valence=float(memory_dict.get("valence", 0.0)),
            intensity=float(memory_dict.get("intensity", 0.0)),
            emotion=str(memory_dict.get("emotion", "neutral")),
        )

    def should_protect_from_decay(self, valence: EmotionalValence) -> bool:
        """
        Determine if a memory should be shielded from baseline heuristic decay.
        High-valence memories are decay-immune — they persist like traumatic
        or deeply joyful biological memories.
        """
        return valence.is_significant

    def apply_valence_to_edge(self, session, source: str, rel_type: str,
                               target: str, valence: EmotionalValence):
        """
        Write emotional valence properties onto a graph relationship.
        """
        clean_rel = "".join(c for c in rel_type if c.isalnum() or c == "_").upper()
        if not clean_rel:
            clean_rel = "RELATED_TO"

        query = f"""
        MATCH (a {{name: $source}})-[r:{clean_rel}]->(b {{name: $target}})
        SET r.valence = $valence,
            r.intensity = $intensity,
            r.emotion = $emotion
        RETURN r
        """
        session.run(query, source=source, target=target,
                    valence=valence.valence, intensity=valence.intensity,
                    emotion=valence.emotion)
        logger.info(f"Amygdala tagged: {source}-[{clean_rel}]->{target} with {valence}")

    def get_decay_immunity_cypher(self) -> str:
        """
        Returns a Cypher WHERE clause fragment that protects emotionally
        significant edges from baseline decay.

        This is injected into the neuroplasticity decay query.
        """
        threshold = self.decay_immunity_threshold
        return (
            f"AND (coalesce(abs(s.valence), 0) < {threshold} "
            f"AND coalesce(s.intensity, 0) < {threshold})"
        )

    def tag_memories(self, driver, memories: List[Dict[str, Any]]):
        """
        Batch-apply emotional valence to a list of extracted memories.
        Each memory dict should contain source, relation, target, and
        optionally valence/intensity/emotion.
        """
        if not driver:
            return

        with driver.session() as session:
            for mem in memories:
                source = mem.get("source")
                rel = mem.get("relation")
                target = mem.get("target")

                if not all([source, rel, target]):
                    continue

                valence = self.extract_valence(mem)
                if valence.emotion != "neutral" or valence.valence != 0.0:
                    try:
                        self.apply_valence_to_edge(session, source, rel, target, valence)
                    except Exception as e:
                        logger.warning(f"Failed to tag valence on {source}->{target}: {e}")
