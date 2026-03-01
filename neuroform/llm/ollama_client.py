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

    def chat_with_memory(self, user_id: str, message: str) -> str:
        """
        Processes a user message, fetching context from Neo4j,
        querying Ollama, and saving extracted facts back to Neo4j.
        Uses WorkingMemory as the prefrontal cortex attention buffer.
        """
        # 1. Record the user message in working memory
        self.working_memory.add_conversation_turn("user", message)

        # 2. Fetch memory context from graph and inject into working memory
        try:
            context_data = self.kg.query_context("User", layer=GraphLayer.NARRATIVE)
            if context_data:
                self.working_memory.add_graph_context(context_data)

            # 3. Build context from working memory's attention-gated buffer
            context_str = self.working_memory.build_context_string()

            # 4. Construct Prompt for Ollama
            system_prompt = f"""You are NeuroForm, an autonomous AI with a multi-layered Neo4j memory system.
Your goal is to answer the user gracefully.

[SRC:KG]
{context_str}

CRITICAL: If you learn a new fact from the user in this turn, you MUST output a JSON block at the very end of your response inside ```json tags.
Format:
```json
{{
    "new_memories": [
        {{"source": "User", "relation": "LIKES", "target": "Apples", "layer": "SOCIAL",
         "valence": 0.5, "intensity": 0.3, "emotion": "joy"}}
    ]
}}
```
{self.amygdala.VALENCE_EXTRACTION_PROMPT}
Only do this if a clear, long-term fact is declared. Otherwise omit the JSON block.
"""

            # 5. Build messages with conversation history from working memory
            messages = [{"role": "system", "content": system_prompt}]
            for turn in self.working_memory.get_conversation_history()[-6:]:
                messages.append(turn)

            response = ollama.chat(model=self.model, messages=messages)
            
            reply_text = response['message']['content']

            # 6. Record the assistant response in working memory
            self.working_memory.add_conversation_turn("assistant", reply_text)

            # 7. Extract and save memories (with amygdala emotional tagging)
            self._extract_and_save_memories(reply_text)
            
            # Clean up JSON block from user visibility (optional but good UX)
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

