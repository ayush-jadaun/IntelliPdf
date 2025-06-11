# IntelliPDF API Documentation

## Overview

The IntelliPDF API provides a comprehensive set of endpoints for document intelligence, semantic search, knowledge graph generation, and AI-powered insights. Built with FastAPI, it offers automatic documentation, high performance, and production-ready features.

**Base URL**: `http://localhost:8000/api/v1`

## Authentication

Currently, the API uses simple API key authentication for development. Production deployments should implement OAuth2 or JWT tokens.

```bash
# Include API key in headers
Authorization: Bearer YOUR_API_KEY
```

## Core Endpoints

### Document Management

#### Upload Documents
```http
POST /documents/upload
```

Upload single or multiple PDF documents for processing.

**Request:**
```bash
curl -X POST \
  -H "Content-Type: multipart/form-data" \
  -F "files=@document1.pdf" \
  -F "files=@document2.pdf" \
  http://localhost:8000/api/v1/documents/upload
```

**Response:**
```json
{
  "status": "success",
  "documents": [
    {
      "id": "doc_123456",
      "filename": "document1.pdf",
      "size": 1024000,
      "pages": 25,
      "status": "processing",
      "uploaded_at": "2025-06-11T10:30:00Z"
    }
  ],
  "processing_jobs": ["job_789012", "job_345678"]
}
```

#### Get Document Status
```http
GET /documents/{document_id}/status
```

Check processing status of uploaded documents.

**Response:**
```json
{
  "id": "doc_123456",
  "status": "completed",
  "progress": 100,
  "processing_steps": {
    "text_extraction": "completed",
    "embedding_generation": "completed",
    "entity_recognition": "completed",
    "graph_construction": "completed"
  },
  "metadata": {
    "total_chunks": 156,
    "entities_found": 45,
    "concepts_extracted": 28
  }
}
```

#### List Documents
```http
GET /documents
```

Retrieve all uploaded documents with metadata.

**Query Parameters:**
- `page` (int): Page number (default: 1)
- `size` (int): Items per page (default: 20)
- `status` (str): Filter by status (processing, completed, failed)
- `search` (str): Search in document titles/content

**Response:**
```json
{
  "documents": [
    {
      "id": "doc_123456",
      "filename": "AI_Research_Paper.pdf",
      "title": "Advances in Neural Network Architecture",
      "authors": ["Dr. Smith", "Prof. Johnson"],
      "pages": 25,
      "size": 1024000,
      "status": "completed",
      "uploaded_at": "2025-06-11T10:30:00Z",
      "concepts": ["neural networks", "deep learning", "AI"],
      "similarity_score": 0.95
    }
  ],
  "total": 150,
  "page": 1,
  "size": 20
}
```

### Semantic Search

#### Search Documents
```http
POST /search/semantic
```

Perform semantic search across document library.

**Request:**
```json
{
  "query": "machine learning applications in healthcare",
  "limit": 10,
  "filters": {
    "document_ids": ["doc_123", "doc_456"],
    "date_range": {
      "start": "2020-01-01",
      "end": "2025-12-31"
    },
    "concepts": ["healthcare", "AI"]
  },
  "include_snippets": true
}
```

**Response:**
```json
{
  "results": [
    {
      "document_id": "doc_123456",
      "document_title": "AI in Medical Diagnosis",
      "similarity_score": 0.92,
      "chunks": [
        {
          "chunk_id": "chunk_789",
          "content": "Machine learning algorithms have shown remarkable success in medical imaging...",
          "page_number": 5,
          "score": 0.94
        }
      ],
      "entities": ["medical imaging", "diagnosis", "accuracy"],
      "snippet": "...remarkable success in medical imaging applications, achieving 95% accuracy..."
    }
  ],
  "total_results": 45,
  "query_time_ms": 120
}
```

#### Search Similar Documents
```http
GET /search/similar/{document_id}
```

Find documents similar to a specific document.

**Query Parameters:**
- `limit` (int): Number of results (default: 10)
- `threshold` (float): Similarity threshold (default: 0.7)

### Knowledge Graph

#### Get Graph Data
```http
GET /graph/data
```

Retrieve knowledge graph nodes and edges.

**Query Parameters:**
- `center_document` (str): Focus on specific document
- `max_depth` (int): Graph traversal depth (default: 2)
- `min_similarity` (float): Minimum edge weight (default: 0.5)

**Response:**
```json
{
  "nodes": [
    {
      "id": "doc_123456",
      "type": "document",
      "title": "AI in Healthcare",
      "concepts": ["AI", "healthcare", "diagnosis"],
      "size": 25,
      "cluster": "healthcare_ai"
    },
    {
      "id": "concept_ai",
      "type": "concept",
      "label": "Artificial Intelligence",
      "frequency": 156,
      "documents": ["doc_123", "doc_456"]
    }
  ],
  "edges": [
    {
      "source": "doc_123456",
      "target": "doc_789012",
      "weight": 0.85,
      "type": "similarity",
      "shared_concepts": ["AI", "neural networks"]
    }
  ],
  "clusters": [
    {
      "id": "healthcare_ai",
      "label": "Healthcare AI",
      "color": "#FF6B6B",
      "documents": 12
    }
  ]
}
```

#### Get Document Connections
```http
GET /graph/connections/{document_id}
```

Get all connections for a specific document.

### AI Chat Interface

#### Start Chat Session
```http
POST /chat/sessions
```

Create a new chat session for document exploration.

**Request:**
```json
{
  "context_documents": ["doc_123", "doc_456"],
  "session_name": "Healthcare AI Research"
}
```

**Response:**
```json
{
  "session_id": "chat_session_789",
  "created_at": "2025-06-11T10:30:00Z",
  "context_summary": "2 documents loaded focusing on healthcare AI applications"
}
```

#### Send Chat Message
```http
POST /chat/sessions/{session_id}/messages
```

Send a message to the AI assistant.

**Request:**
```json
{
  "message": "What are the main trends in AI diagnostic tools?",
  "include_sources": true,
  "search_scope": "all" // or "context_documents"
}
```

**Response:**
```json
{
  "message_id": "msg_456789",
  "response": "Based on the research papers in your library, there are three main trends in AI diagnostic tools:\n\n1. **Deep Learning for Medical Imaging**: Convolutional neural networks are achieving human-level accuracy...",
  "sources": [
    {
      "document_id": "doc_123456",
      "title": "AI in Medical Diagnosis",
      "snippet": "CNNs achieving 95% accuracy in radiology...",
      "page": 12,
      "relevance_score": 0.91
    }
  ],
  "suggested_followups": [
    "What specific imaging techniques benefit most from AI?",
    "How do these diagnostic tools handle edge cases?"
  ],
  "processing_time_ms": 1500
}
```

### AI Insights

#### Generate Document Insights
```http
POST /insights/generate
```

Generate AI-powered insights for documents or document collections.

**Request:**
```json
{
  "document_ids": ["doc_123", "doc_456"],
  "insight_types": ["summary", "trends", "gaps", "recommendations"],
  "depth": "detailed" // or "brief"
}
```

**Response:**
```json
{
  "insights": {
    "summary": {
      "key_findings": [
        "AI diagnostic tools show 15% improvement over traditional methods",
        "Computer vision applications dominate current research"
      ],
      "main_themes": ["accuracy", "efficiency", "clinical adoption"]
    },
    "trends": [
      {
        "trend": "Increasing use of transformer architectures",
        "confidence": 0.87,
        "supporting_documents": 8,
        "time_evolution": "2022-2025"
      }
    ],
    "research_gaps": [
      {
        "gap": "Limited studies on AI explainability in clinical settings",
        "potential_impact": "high",
        "suggested_research": "Develop interpretable AI models for medical diagnosis"
      }
    ],
    "recommendations": [
      {
        "recommendation": "Focus research on multimodal diagnostic approaches",
        "rationale": "Only 15% of papers explore combining imaging with clinical data",
        "priority": "high"
      }
    ]
  },
  "generation_time_ms": 3500
}
```

#### Get Research Recommendations
```http
GET /insights/recommendations/{document_id}
```

Get personalized research recommendations based on a document.

## Batch Processing

#### Batch Upload
```http
POST /batch/upload
```

Upload and process large document collections.

**Request:**
```json
{
  "documents": [
    {"url": "https://example.com/paper1.pdf"},
    {"file_path": "/tmp/paper2.pdf"}
  ],
  "processing_options": {
    "extract_tables": true,
    "ocr_enabled": true,
    "language": "en"
  },
  "callback_url": "https://yourapp.com/webhook/batch-complete"
}
```

#### Get Batch Status
```http
GET /batch/jobs/{job_id}
```

## WebSocket Endpoints

### Real-time Processing Updates
```
ws://localhost:8000/ws/processing/{document_id}
```

Receive real-time updates during document processing.

**Message Format:**
```json
{
  "type": "processing_update",
  "document_id": "doc_123456",
  "step": "embedding_generation",
  "progress": 45,
  "estimated_time_remaining": 120
}
```

### Live Chat
```
ws://localhost:8000/ws/chat/{session_id}
```

Real-time chat interface with streaming responses.

## Error Handling

All endpoints return consistent error responses:

```json
{
  "error": {
    "code": "DOCUMENT_NOT_FOUND",
    "message": "Document with ID doc_123456 not found",
    "details": {
      "document_id": "doc_123456",
      "timestamp": "2025-06-11T10:30:00Z"
    }
  }
}
```

### Error Codes

- `DOCUMENT_NOT_FOUND`: Document doesn't exist
- `PROCESSING_FAILED`: Document processing error
- `INVALID_FILE_FORMAT`: Unsupported file type
- `QUOTA_EXCEEDED`: API rate limit reached
- `INVALID_PARAMETERS`: Request validation failed
- `SERVICE_UNAVAILABLE`: External service error

## Rate Limiting

API endpoints are rate-limited to ensure fair usage:

- **Free Tier**: 100 requests/hour
- **Pro Tier**: 1000 requests/hour
- **Enterprise**: Custom limits

Rate limit headers are included in responses:
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1625097600
```

## SDKs and Client Libraries

### Python SDK
```python
from intellipdf import IntelliPDFClient

client = IntelliPDFClient(api_key="your_api_key")

# Upload document
result = client.documents.upload("research_paper.pdf")

# Search
results = client.search.semantic("machine learning applications")

# Chat
response = client.chat.ask("What are the main findings?")
```

### JavaScript SDK
```javascript
import { IntelliPDFClient } from '@intellipdf/sdk';

const client = new IntelliPDFClient({ apiKey: 'your_api_key' });

// Upload document
const result = await client.documents.upload(file);

// Search
const results = await client.search.semantic('machine learning applications');
```

## Production Considerations

### Performance
- Responses cached for 5 minutes by default
- Document processing is asynchronous
- Vector search optimized for sub-second response times

### Security
- API keys should be kept secure
- HTTPS required in production
- File uploads are virus-scanned
- Input validation on all endpoints

### Monitoring
- Health check endpoint: `GET /health`
- Metrics endpoint: `GET /metrics`
- OpenAPI documentation: `GET /docs`