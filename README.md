<div align="center">

# Rag Knowledge Graph MCP

**MCP server for rag knowledge graph mcp operations**

[![PyPI](https://img.shields.io/pypi/v/meok-rag-knowledge-graph-mcp)](https://pypi.org/project/meok-rag-knowledge-graph-mcp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![MEOK AI Labs](https://img.shields.io/badge/MEOK_AI_Labs-MCP_Server-purple)](https://meok.ai)

</div>

## Overview

Rag Knowledge Graph MCP provides AI-powered tools via the Model Context Protocol (MCP).

## Tools

| Tool | Description |
|------|-------------|
| `index_document` | Index a document for RAG retrieval. Generates embeddings and extracts entities. |
| `rag_query` | Query the knowledge base. Methods: vector (semantic), keyword (FTS5), hybrid (bo |
| `add_graph_edge` | Add a relationship to the knowledge graph. |
| `graph_query` | Traverse the knowledge graph from an entity to find connections. |
| `get_knowledge_stats` | Get knowledge base statistics. |

## Installation

```bash
pip install meok-rag-knowledge-graph-mcp
```

## Usage with Claude Desktop

Add to your Claude Desktop MCP config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "rag-knowledge-graph": {
      "command": "python",
      "args": ["-m", "meok_rag_knowledge_graph_mcp.server"]
    }
  }
}
```

## Usage with FastMCP

```python
from mcp.server.fastmcp import FastMCP

# This server exposes 5 tool(s) via MCP
# See server.py for full implementation
```

## License

MIT © [MEOK AI Labs](https://meok.ai)
