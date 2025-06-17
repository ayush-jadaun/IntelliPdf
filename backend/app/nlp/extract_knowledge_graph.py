import spacy
from typing import List, Dict, Any
from app.schemas.document import KnowledgeGraph

nlp = spacy.load("en_core_web_sm")

def extract_knowledge_graph_from_text(text: str) -> KnowledgeGraph:
    doc = nlp(text)
    nodes = []
    edges = []

    entity_map = {}

    # Step 1: Extract named entities as nodes
    for ent in doc.ents:
        node_id = f"{ent.label_}_{ent.text}"
        if node_id not in entity_map:
            entity_map[node_id] = {
                "id": node_id,
                "label": ent.text,
                "type": ent.label_,
                "score": None
            }
            nodes.append(entity_map[node_id])

    # Step 2: Naive co-occurrence-based relation (can be improved)
    for sent in doc.sents:
        ents = [ent for ent in sent.ents]
        for i in range(len(ents)):
            for j in range(i+1, len(ents)):
                source = f"{ents[i].label_}_{ents[i].text}"
                target = f"{ents[j].label_}_{ents[j].text}"
                edge = {
                    "source": source,
                    "target": target,
                    "type": "CO_OCCURRENCE",
                    "weight": 1.0
                }
                edges.append(edge)
    print("✅ Extracted nodes:", nodes)
    print("✅ Extracted edges:", edges)
    return KnowledgeGraph(nodes=nodes, edges=edges)
