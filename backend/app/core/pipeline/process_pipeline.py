from app.schemas.document import ProcessedDocumentResponse
from app.core.database.knowledge_graph import store_knowledge_graph
from app.nlp.extract_knowledge_graph import extract_knowledge_graph_from_text

def process_pipeline(document_data: dict):
    parsed_doc = ProcessedDocumentResponse(**document_data)

    if not parsed_doc.knowledge_graph or not parsed_doc.knowledge_graph.nodes:
        # Automatically extract if not already present
        parsed_doc.knowledge_graph = extract_knowledge_graph_from_text(parsed_doc.full_text)

    if parsed_doc.knowledge_graph:
        store_knowledge_graph(parsed_doc.knowledge_graph)
