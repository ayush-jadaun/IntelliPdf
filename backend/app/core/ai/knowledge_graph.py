"""
Knowledge Graph Construction Utilities for IntelliPDF

This module provides functions to build a simple knowledge graph (KG) from
entities and keywords extracted from document text analytics, suitable for
visualization, querying, or exporting to a graph database.

Nodes:
    - Documents
    - Entities (PERSON, ORG, GPE, etc. from NER)
    - Keywords

Edges:
    - Document-to-entity (e.g., "has_person", "has_org")
    - Document-to-keyword ("has_keyword")
    - Optional: entity-to-entity co-occurrence, keyword-to-entity, etc.

Author: IntelliPDF Team
"""

from typing import Dict, List, Any, Optional
import hashlib

def _hash_id(*args) -> str:
    """Generate a unique node id by hashing its identifying fields."""
    base_str = "|".join(str(a) for a in args)
    return hashlib.sha1(base_str.encode()).hexdigest()[:16]

def build_knowledge_graph(
    entities: Dict[str, List[str]],
    keywords: Dict[str, float],
    doc_metadata: Dict[str, Any],
    doc_id: Optional[str] = None
) -> Dict[str, List[Dict]]:
    """
    Build a simple knowledge graph representation from extracted entities and keywords for a single document.
    Returns a dict with 'nodes' and 'edges'.
    """
    nodes = []
    edges = []
    node_ids = {}

    # Document node
    doc_title = doc_metadata.get("title") or doc_id or doc_metadata.get("file_path", "unknown_doc")
    doc_node_id = f"doc:{_hash_id(doc_title)}"
    nodes.append({"id": doc_node_id, "label": doc_title, "type": "document"})
    node_ids[doc_node_id] = len(nodes) - 1

    # Entity nodes and edges
    for ent_type, ent_list in entities.items():
        for ent in set(ent_list):
            ent_str = str(ent).strip()
            if not ent_str:
                continue
            ent_node_id = f"{ent_type}:{_hash_id(ent_type, ent_str)}"
            if ent_node_id not in node_ids:
                nodes.append({"id": ent_node_id, "label": ent_str, "type": ent_type})
                node_ids[ent_node_id] = len(nodes) - 1
            edges.append({
                "source": doc_node_id,
                "target": ent_node_id,
                "type": f"has_{ent_type.lower()}"
            })

    # Keyword nodes and edges
    for kw, score in keywords.items():
        kw_str = str(kw).strip()
        if not kw_str:
            continue
        kw_node_id = f"kw:{_hash_id(kw_str)}"
        if kw_node_id not in node_ids:
            nodes.append({"id": kw_node_id, "label": kw_str, "type": "keyword", "score": float(score)})
            node_ids[kw_node_id] = len(nodes) - 1
        edges.append({
            "source": doc_node_id,
            "target": kw_node_id,
            "type": "has_keyword",
            "weight": float(score)
        })

    # (Optional) Entity co-occurrence edges (within the same doc)
    # For each pair of entity types, link if they appear together.
    entity_types = list(entities.keys())
    for i, type1 in enumerate(entity_types):
        for type2 in entity_types[i + 1:]:
            for ent1 in set(entities[type1]):
                for ent2 in set(entities[type2]):
                    ent1_id = f"{type1}:{_hash_id(type1, ent1)}"
                    ent2_id = f"{type2}:{_hash_id(type2, ent2)}"
                    if ent1_id != ent2_id and ent1_id in node_ids and ent2_id in node_ids:
                        # Only add if not self-loop and both exist
                        edges.append({
                            "source": ent1_id,
                            "target": ent2_id,
                            "type": "co_occurs"
                        })

    return {"nodes": nodes, "edges": edges}

def merge_knowledge_graphs(graphs: List[Dict[str, List[Dict]]]) -> Dict[str, List[Dict]]:
    """
    Merge multiple single-document KGs into one global KG.
    Handles deduplication of nodes by id.
    """
    all_nodes = {}
    all_edges = []

    for graph in graphs:
        # Merge nodes
        for node in graph.get("nodes", []):
            all_nodes[node["id"]] = node
        # Merge edges (edges can be duplicated, that's OK for most graph DBs/visualizers)
        all_edges.extend(graph.get("edges", []))

    return {"nodes": list(all_nodes.values()), "edges": all_edges}