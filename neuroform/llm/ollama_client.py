import logging
from typing import Dict, Any, List
import json
import ollama
from neuroform.memory.graph import KnowledgeGraph, GraphLayer

logger = logging.getLogger(__name__)

class OllamaClient:
    def __init__(self, kg: KnowledgeGraph, model: str = "llama3"):
        self.kg = kg
        self.model = model

    def chat_with_memory(self, user_id: str, message: str) -> str:
        """
        Processes a user message, fetching context from Neo4j,
        querying Ollama, and saving extracted facts back to Neo4j.
        """
        # 1. Fetch memory context
        try:
            # We fetch context based on the user's name or keywords.
            # Simplified for standalone: just fetch NARRATIVE layer related to "User".
            context_data = self.kg.query_context("User", layer=GraphLayer.NARRATIVE)
            
            # Format context
            context_str = "No prior memory context."
            if context_data:
                context_str = "Prior Memory:\\n"
                for item in context_data:
                    context_str += f"- {item['source']} ({item['relationship']}) {item['target']}\\n"

            # 2. Construct Prompt for Ollama
            system_prompt = f"""You are NeuroForm, an autonomous AI with a multi-layered Neo4j memory system.
Your goal is to answer the user gracefully.

[SRC:KG]
{context_str}

CRITICAL: If you learn a new fact from the user in this turn, you MUST output a JSON block at the very end of your response inside ```json tags.
Format:
```json
{{
    "new_memories": [
        {{"source": "User", "relation": "LIKES", "target": "Apples", "layer": "SOCIAL"}}
    ]
}}
```
Only do this if a clear, long-term fact is declared. Otherwise omit the JSON block.
"""

            response = ollama.chat(model=self.model, messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ])
            
            reply_text = response['message']['content']

            # 3. Extract and save memories
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
                    
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON memory block from LLM output.")
        except Exception as e:
            logger.error(f"Error saving memories: {e}")
