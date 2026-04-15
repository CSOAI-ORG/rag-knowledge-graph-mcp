# Rag Knowledge Graph

> By [MEOK AI Labs](https://meok.ai) — MEOK AI Labs — RAG + Knowledge Graph. Hybrid retrieval: vector embeddings + graph relationships + hierarchical navigation.

RAG Knowledge Graph MCP — MEOK AI Labs. Vector search + knowledge graph + unified context retrieval.

## Installation

```bash
pip install rag-knowledge-graph-mcp
```

## Usage

```bash
# Run standalone
python server.py

# Or via MCP
mcp install rag-knowledge-graph-mcp
```

## Tools

### `index_document`
Index a document for RAG retrieval. Generates embeddings and extracts entities.

**Parameters:**
- `content` (str)
- `metadata` (str)
- `doc_id` (str)

### `rag_query`
Query the knowledge base. Methods: vector (semantic), keyword (FTS5), hybrid (both), graph (relationship traversal).

**Parameters:**
- `query` (str)
- `top_k` (int)
- `method` (str)

### `add_graph_edge`
Add a relationship to the knowledge graph.

**Parameters:**
- `source_name` (str)
- `target_name` (str)
- `relation` (str)
- `weight` (float)

### `graph_query`
Traverse the knowledge graph from an entity to find connections.

**Parameters:**
- `entity_name` (str)
- `depth` (int)

### `get_knowledge_stats`
Get knowledge base statistics.


## Authentication

Free tier: 15 calls/day. Upgrade at [meok.ai/pricing](https://meok.ai/pricing) for unlimited access.

## Links

- **Website**: [meok.ai](https://meok.ai)
- **GitHub**: [CSOAI-ORG/rag-knowledge-graph-mcp](https://github.com/CSOAI-ORG/rag-knowledge-graph-mcp)
- **PyPI**: [pypi.org/project/rag-knowledge-graph-mcp](https://pypi.org/project/rag-knowledge-graph-mcp/)

## License

MIT — MEOK AI Labs
