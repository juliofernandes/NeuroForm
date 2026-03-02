import logging
import os
from neo4j import GraphDatabase, Driver
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class GraphLayer:
    NARRATIVE = "NARRATIVE"
    SEMANTIC = "SEMANTIC"
    EPISODIC = "EPISODIC"
    SOCIAL = "SOCIAL"
    SYSTEM = "SYSTEM"
    PROCEDURAL = "PROCEDURAL"

class KnowledgeGraph:
    def __init__(self, uri: str = None, user: str = None, password: str = None):
        self.uri = uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.environ.get("NEO4J_USER", "neo4j")
        self.password = password or os.environ.get("NEO4J_PASSWORD", "password")
        self.driver: Optional[Driver] = None
        
        self.connect()

    def connect(self):
        if os.environ.get("DISABLE_NEO4J") == "true":
            logger.warning("Neo4j is disabled via DISABLE_NEO4J env.")
            return

        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self.driver.verify_connectivity()
            logger.info("Connected to Neo4j successfully.")
            self._initialize_schema()
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self.driver = None

    def _initialize_schema(self):
        if not self.driver:
            return
            
        queries = [
            "CREATE INDEX node_name_idx IF NOT EXISTS FOR (n:Entity) ON (n.name)",
            "CREATE INDEX node_layer_idx IF NOT EXISTS FOR (n:Entity) ON (n.layer)",
            # Per-user per-scope indexes — ground rule
            "CREATE INDEX node_uid_idx IF NOT EXISTS FOR (n:Entity) ON (n.user_id)",
            "CREATE INDEX node_scope_idx IF NOT EXISTS FOR (n:Entity) ON (n.scope)",
        ]
        
        # Create layer-specific indexes
        layers = [GraphLayer.NARRATIVE, GraphLayer.SEMANTIC, GraphLayer.EPISODIC, 
                  GraphLayer.SOCIAL, GraphLayer.SYSTEM, GraphLayer.PROCEDURAL]
                  
        for layer in layers:
            queries.append(f"CREATE INDEX {layer.lower()}_layer_idx IF NOT EXISTS FOR (n:{layer}) ON (n.layer)")

        with self.driver.session() as session:
            for q in queries:
                session.run(q)

    def close(self):
        if self.driver:
            self.driver.close()
            self.driver = None

    def clear_all(self) -> int:
        if not self.driver:
            return 0
        with self.driver.session() as session:
            result = session.run("MATCH (n) DETACH DELETE n")
            summary = result.consume()
            return summary.counters.nodes_deleted

    def ensure_layer_root(self, layer: str):
        """Ensures a root node exists for the layer and is connected to existing layer roots."""
        if not self.driver:
            return
            
        query = """
        MERGE (root:LayerRoot {name: $layer, type: 'root'})
        WITH root
        OPTIONAL MATCH (other:LayerRoot) WHERE other.name <> root.name
        WITH root, collect(other) as others
        FOREACH (o IN others | MERGE (root)-[:PEER_LAYER]-(o))
        """
        with self.driver.session() as session:
            session.run(query, layer=layer)

    def add_node(self, label: str, name: str, layer: str = GraphLayer.NARRATIVE,
                 properties: Dict[str, Any] = None,
                 user_id: str = "", scope: str = "PUBLIC"):
        """
        Add or merge a node. Every node carries user_id + scope as ground-rule
        properties for per-user per-scope isolation.
        """
        if not self.driver:
            return
        
        props = properties or {}
        props["name"] = name
        props["layer"] = layer
        props["user_id"] = user_id
        props["scope"] = scope
        
        # Construct SET clause from dictionary
        set_clauses = []
        params = {"name": name, "layer": layer, "user_id": user_id, "scope": scope}
        for k, v in props.items():
            if k not in ["name", "layer", "user_id", "scope"]:
                set_clauses.append(f"n.{k} = ${k}")
                params[k] = v
                
        set_query = ""
        if set_clauses:
            set_query = "SET " + ", ".join(set_clauses)

        self.ensure_layer_root(layer)

        query = f"""
        MATCH (root:LayerRoot {{name: $layer}})
        MERGE (n:{label} {{name: $name, layer: $layer}})
        SET n.user_id = $user_id, n.scope = $scope
        {set_query}
        MERGE (n)-[:IN_LAYER]->(root)
        SET n.last_fired = timestamp()
        RETURN n
        """
        
        with self.driver.session() as session:
            session.run(query, **params)

    def add_relationship(self, source_name: str, rel_type: str, target_name: str,
                         strength: float = 1.0,
                         user_id: str = "", scope: str = "PUBLIC"):
        """
        Add or merge a relationship. Every edge carries user_id + scope as
        ground-rule properties for per-user per-scope isolation.
        """
        if not self.driver:
            return
            
        # Sanitize rel_type (alphanumeric and underscores only)
        clean_rel_type = "".join(c for c in rel_type if c.isalnum() or c == "_").upper()
        if not clean_rel_type:
            clean_rel_type = "RELATED_TO"

        query = f"""
        MATCH (a {{name: $source}}), (b {{name: $target}})
        MERGE (a)-[r:{clean_rel_type}]->(b)
        ON CREATE SET r.strength = $strength, r.created = timestamp(),
                      r.last_fired = timestamp(),
                      r.user_id = $user_id, r.scope = $scope
        ON MATCH SET r.strength = r.strength + ($strength * 0.1),
                     r.last_fired = timestamp(),
                     r.user_id = $user_id, r.scope = $scope
        RETURN r
        """
        with self.driver.session() as session:
            session.run(query, source=source_name, target=target_name,
                        strength=strength, user_id=user_id, scope=scope)

    def query_context(self, entity_name: str, layer: Optional[str] = None,
                      user_id: str = "", scope: str = "PUBLIC") -> List[Dict[str, Any]]:
        """
        Query context with per-user per-scope isolation.

        Visibility rules:
        - PUBLIC scope: see PUBLIC nodes/edges only
        - PRIVATE scope + user_id: see (user_id's PRIVATE) + all PUBLIC
        - No user_id: PUBLIC only
        """
        if not self.driver:
            return []
            
        layer_filter = "AND a.layer = $layer" if layer else ""

        # Scope isolation: show PUBLIC always, show PRIVATE only for matching user
        if user_id and scope in ("PRIVATE", "CORE_PRIVATE"):
            scope_filter = (
                "AND ((r.scope = 'PUBLIC' OR r.scope IS NULL) "
                "OR (r.user_id = $user_id))"
            )
        else:
            scope_filter = "AND (r.scope = 'PUBLIC' OR r.scope IS NULL)"

        query = f"""
        MATCH (a {{name: $name}})-[r]-(b)
        WHERE 1=1 {layer_filter} {scope_filter}
        SET r.last_fired = timestamp()
        RETURN a.name AS a_name, a.layer AS a_layer, type(r) AS rel,
               r.strength AS strength, b.name AS b_name, b.layer AS b_layer,
               r.user_id AS r_user_id, r.scope AS r_scope
        ORDER BY r.strength DESC
        LIMIT 25
        """
        
        params = {"name": entity_name, "user_id": user_id}
        if layer:
            params["layer"] = layer
            
        results = []
        with self.driver.session() as session:
            records = session.run(query, **params)
            for record in records:
                results.append({
                    "source": record["a_name"],
                    "source_layer": record["a_layer"],
                    "relationship": record["rel"],
                    "strength": record["strength"],
                    "target": record["b_name"],
                    "target_layer": record["b_layer"],
                    "user_id": record["r_user_id"] or "",
                    "scope": record["r_scope"] or "PUBLIC",
                })
        return results
