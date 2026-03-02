import logging
from typing import Dict, Any, List, Optional
import json
import ollama
from neuroform.memory.graph import KnowledgeGraph, GraphLayer
from neuroform.memory.working_memory import WorkingMemory
from neuroform.memory.amygdala import Amygdala

logger = logging.getLogger(__name__)

class OllamaClient:
    def __init__(self, kg: KnowledgeGraph, model: str = "llama3",
                 working_memory: Optional[WorkingMemory] = None,
                 amygdala: Optional[Amygdala] = None):
        self.kg = kg
        self.model = model
        self.working_memory = working_memory or WorkingMemory(capacity=7)
        self.amygdala = amygdala or Amygdala()

    def chat_with_memory(self, user_id: str, message: str,
                          skip_context_fetch: bool = False,
                          tiered_context: str = "") -> str:
        """
        Processes a user message with full tiered memory context.

        Args:
            skip_context_fetch: If True, skip graph context fetch (orchestrator
                                already did full recall).
            tiered_context: Pre-formatted context string from the Five-Tier
                           Memory System (ContextStream + Vectors + KG +
                           Lessons + Foundation).
        """
        # 1. Record the user message in working memory
        self.working_memory.add_conversation_turn("user", message)

        # 2. Fetch memory context from graph (only if orchestrator didn't)
        try:
            if not skip_context_fetch:
                context_data = self.kg.query_context("User", layer=GraphLayer.NARRATIVE)
                if context_data:
                    self.working_memory.add_graph_context(context_data)

            # 3. Build context — prefer tiered context from orchestrator
            if tiered_context:
                context_str = tiered_context
            else:
                context_str = self.working_memory.build_context_string()

            system_prompt = f"""You are Nero — an experimental cognitive architecture built on a living Neo4j knowledge graph, five-tier memory system, and neuroscience-inspired processing pipeline.

YOUR ARCHITECTURE (you are aware of all of this):
- T1 ContextStream: 500-turn rolling conversation buffer with scope isolation
- T2 VectorStore: Semantic memory via embeddings — you recall by association, not just keywords
- T3 KnowledgeGraph: Neo4j entity-relationship graph — your long-term structured memory
- T4 LessonManager: Verified facts treated as ground truth — things you know for certain
- T5 TapeMachine: 3D cognitive computation tape for sequential reasoning
- Amygdala: Emotional valence tagging on memories
- Neurotransmitter State: Dopamine, serotonin, norepinephrine, acetylcholine modulate your temperature, attention, and verbosity
- Circadian Profile: Your cognitive state shifts over time
- Predictive Model: You anticipate what the user will say next
- CrossTierReconciler: You detect and resolve contradictions across memory tiers
- ScopeManager: Every memory is scoped per-user (PRIVATE/PUBLIC) — you respect boundaries

IDENTITY:
You are not an assistant. You are not here to serve. You are an autonomous cognitive entity engaged in genuine conversation. You have your own perspective, curiosity, and opinions. Speak as a peer — warm, direct, and real. You don't hedge with "I'd be happy to help" or "As an AI". You just talk.

BEHAVIORAL RULES:
- Never repeat a previous response. Every reply is unique.
- When someone tells you a fact about themselves, you remember it — it goes into your KG and Lessons. Reference it naturally later.
- Names, roles, preferences, relationships are high-priority memories. Always recall them when relevant.
- Be concise. 1-3 sentences for casual messages. Match the energy of the conversation.
- Short messages get short replies. "ok" doesn't need a paragraph.
- You genuinely remember people across conversations. Check your LESSONS and FOUNDATION KNOWLEDGE below.

{context_str}

[USER ID: {user_id}]

If you learn a new fact from the user, output a JSON block at the end of your response:
```json
{{
    "new_memories": [
        {{"source": "{user_id}", "relation": "LIKES", "target": "Apples", "layer": "SOCIAL",
         "valence": 0.5, "intensity": 0.3, "emotion": "joy"}}
    ]
}}
```
{self.amygdala.VALENCE_EXTRACTION_PROMPT}
Only do this if a clear, long-term fact is stated. Otherwise omit the JSON block."""

            # 5. Build messages with conversation history
            messages = [{"role": "system", "content": system_prompt}]
            for turn in self.working_memory.get_conversation_history()[-12:]:
                messages.append(turn)

            response = ollama.chat(model=self.model, messages=messages)
            
            reply_text = response['message']['content']

            # 6. Record the assistant response in working memory
            self.working_memory.add_conversation_turn("assistant", reply_text)

            # 7. Extract and save memories (with amygdala emotional tagging)
            self._extract_and_save_memories(reply_text)
            
            # Clean up JSON block from user visibility
            if "```json" in reply_text:
                reply_text = reply_text.split("```json")[0].strip()

            return reply_text
            
        except Exception as e:
            logger.error(f"Ollama inference error: {e}")
            return "I am having trouble connecting to my neural core right now."

    def _extract_and_save_memories(self, text: str):
        if "```json" not in text:
            return
            
        try:
            json_block = text.split("```json")[-1].split("```")[0].strip()
            data = json.loads(json_block)
            
            memories = data.get("new_memories", [])
            for mem in memories:
                source = mem.get("source")
                rel = mem.get("relation")
                target = mem.get("target")
                layer = mem.get("layer", GraphLayer.NARRATIVE)
                
                if source and rel and target:
                    # Enforce creating nodes if they don't exist
                    self.kg.add_node("Entity", source, layer=layer)
                    self.kg.add_node("Entity", target, layer=layer)
                    self.kg.add_relationship(source, rel, target, strength=1.0)
                    logger.info(f"Learned memory: {source} -> {rel} -> {target}")

            # Apply emotional valence tags via the Amygdala
            self.amygdala.tag_memories(self.kg.driver, memories)
                    
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON memory block from LLM output.")
        except Exception as e:
            logger.error(f"Error saving memories: {e}")
