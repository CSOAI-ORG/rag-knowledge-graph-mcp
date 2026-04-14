#!/usr/bin/env python3
"""RAG Knowledge Graph MCP — MEOK AI Labs. Vector search + knowledge graph + unified context retrieval."""
import json, os, sqlite3, hashlib, math, re
from datetime import datetime, timezone
from typing import Optional
from collections import defaultdict
from mcp.server.fastmcp import FastMCP
from pathlib import Path

FREE_DAILY_LIMIT = 15
_usage = defaultdict(list)
def _rl(c="anon"):
    now = datetime.now(timezone.utc)
    _usage[c] = [t for t in _usage[c] if (now-t).total_seconds() < 86400]
    if len(_usage[c]) >= FREE_DAILY_LIMIT: return json.dumps({"error": f"Limit {FREE_DAILY_LIMIT}/day"})
    _usage[c].append(now); return None

mcp = FastMCP("rag-knowledge-graph", instructions="MEOK AI Labs — RAG + Knowledge Graph. Hybrid retrieval: vector embeddings + graph relationships + hierarchical navigation.")

DB_PATH = Path.home() / ".meok-ai" / "rag-knowledge.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def _get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("CREATE TABLE IF NOT EXISTS documents (id TEXT PRIMARY KEY, content TEXT, metadata TEXT, embedding TEXT, created_at TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS entities (id TEXT PRIMARY KEY, name TEXT, type TEXT, properties TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS edges (source TEXT, target TEXT, relation TEXT, weight REAL DEFAULT 1.0)")
    conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS doc_fts USING fts5(content, metadata)")
    return conn

def _simple_embed(text):
    """TF-IDF-like embedding (no external deps)."""
    words = re.findall(r'\w+', text.lower())
    tf = defaultdict(int)
    for w in words: tf[w] += 1
    total = len(words) or 1
    return {w: round(c/total, 4) for w, c in tf.items()}

def _cosine_sim(a, b):
    common = set(a.keys()) & set(b.keys())
    if not common: return 0.0
    dot = sum(a[k]*b[k] for k in common)
    mag_a = math.sqrt(sum(v**2 for v in a.values()))
    mag_b = math.sqrt(sum(v**2 for v in b.values()))
    return dot / (mag_a * mag_b) if mag_a * mag_b > 0 else 0.0

@mcp.tool()
def index_document(content: str, metadata: str = "", doc_id: str = "") -> str:
    """Index a document for RAG retrieval. Generates embeddings and extracts entities."""
    if err := _rl(): return err
    doc_id = doc_id or hashlib.sha256(content.encode()).hexdigest()[:12]
    embedding = _simple_embed(content)
    conn = _get_db()
    conn.execute("INSERT OR REPLACE INTO documents VALUES (?,?,?,?,?)",
        (doc_id, content, metadata, json.dumps(embedding), datetime.now(timezone.utc).isoformat()))
    try: conn.execute("INSERT INTO doc_fts VALUES (?,?)", (content, metadata))
    except: pass
    # Extract simple entities
    words = set(w.title() for w in re.findall(r'\b[A-Z][a-z]+\b', content) if len(w) > 2)
    for w in list(words)[:20]:
        eid = hashlib.md5(w.encode()).hexdigest()[:8]
        conn.execute("INSERT OR IGNORE INTO entities VALUES (?,?,?,?)", (eid, w, "auto", "{}"))
    conn.commit(); conn.close()
    return json.dumps({"doc_id": doc_id, "indexed": True, "chars": len(content), "entities_extracted": len(words), "embedding_dims": len(embedding)}, indent=2)

@mcp.tool()
def rag_query(query: str, top_k: int = 5, method: str = "hybrid") -> str:
    """Query the knowledge base. Methods: vector (semantic), keyword (FTS5), hybrid (both), graph (relationship traversal)."""
    if err := _rl(): return err
    conn = _get_db()
    results = []
    
    if method in ("keyword", "hybrid"):
        rows = conn.execute("SELECT content, metadata FROM doc_fts WHERE doc_fts MATCH ? LIMIT ?", (query, top_k)).fetchall()
        for content, meta in rows:
            results.append({"content": content[:500], "metadata": meta, "method": "keyword", "score": 0.8})
    
    if method in ("vector", "hybrid"):
        q_embed = _simple_embed(query)
        docs = conn.execute("SELECT id, content, embedding FROM documents").fetchall()
        scored = []
        for did, content, emb_json in docs:
            emb = json.loads(emb_json)
            sim = _cosine_sim(q_embed, emb)
            if sim > 0.05: scored.append((sim, did, content))
        scored.sort(reverse=True)
        for sim, did, content in scored[:top_k]:
            if not any(r["content"][:100] == content[:100] for r in results):
                results.append({"content": content[:500], "doc_id": did, "method": "vector", "score": round(sim, 3)})
    
    if method == "graph":
        q_words = set(w.title() for w in query.split() if len(w) > 2)
        for word in q_words:
            edges = conn.execute("SELECT e.source, e.target, e.relation, n1.name, n2.name FROM edges e JOIN entities n1 ON e.source=n1.id JOIN entities n2 ON e.target=n2.id WHERE n1.name=? OR n2.name=?", (word, word)).fetchall()
            for src, tgt, rel, n1, n2 in edges:
                results.append({"content": f"{n1} --[{rel}]--> {n2}", "method": "graph", "score": 0.7})
    
    conn.close()
    results.sort(key=lambda x: x.get("score", 0), reverse=True)
    return json.dumps({"query": query, "method": method, "results": results[:top_k], "total_found": len(results)}, indent=2)

@mcp.tool()
def add_graph_edge(source_name: str, target_name: str, relation: str, weight: float = 1.0) -> str:
    """Add a relationship to the knowledge graph."""
    if err := _rl(): return err
    conn = _get_db()
    src_id = hashlib.md5(source_name.encode()).hexdigest()[:8]
    tgt_id = hashlib.md5(target_name.encode()).hexdigest()[:8]
    conn.execute("INSERT OR IGNORE INTO entities VALUES (?,?,?,?)", (src_id, source_name, "manual", "{}"))
    conn.execute("INSERT OR IGNORE INTO entities VALUES (?,?,?,?)", (tgt_id, target_name, "manual", "{}"))
    conn.execute("INSERT INTO edges VALUES (?,?,?,?)", (src_id, tgt_id, relation, weight))
    conn.commit(); conn.close()
    return json.dumps({"added": True, "source": source_name, "target": target_name, "relation": relation}, indent=2)

@mcp.tool()
def graph_query(entity_name: str, depth: int = 2) -> str:
    """Traverse the knowledge graph from an entity to find connections."""
    if err := _rl(): return err
    conn = _get_db()
    visited = set()
    results = []
    queue = [(entity_name, 0)]
    while queue:
        name, d = queue.pop(0)
        if name in visited or d > depth: continue
        visited.add(name)
        eid = hashlib.md5(name.encode()).hexdigest()[:8]
        edges = conn.execute("SELECT n2.name, e.relation, e.weight FROM edges e JOIN entities n2 ON e.target=n2.id WHERE e.source=?", (eid,)).fetchall()
        edges += conn.execute("SELECT n1.name, e.relation, e.weight FROM edges e JOIN entities n1 ON e.source=n1.id WHERE e.target=?", (eid,)).fetchall()
        for target, rel, weight in edges:
            results.append({"from": name, "to": target, "relation": rel, "depth": d, "weight": weight})
            if d < depth: queue.append((target, d+1))
    conn.close()
    return json.dumps({"entity": entity_name, "depth": depth, "connections": results, "nodes_visited": len(visited)}, indent=2)

@mcp.tool()
def get_knowledge_stats() -> str:
    """Get knowledge base statistics."""
    conn = _get_db()
    docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    entities = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    edges = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    conn.close()
    return json.dumps({"documents": docs, "entities": entities, "edges": edges, "db_path": str(DB_PATH)}, indent=2)

if __name__ == "__main__":
    mcp.run()
