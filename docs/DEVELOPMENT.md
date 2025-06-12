# IntelliPDF Development Guide

This guide will help you set up the IntelliPDF development environment and start contributing to the project.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Development Environment](#development-environment)
- [Running the Application](#running-the-application)
- [API Development](#api-development)
- [Frontend Development](#frontend-development)
- [Database Development](#database-development)
- [AI/ML Components](#aiml-components)
- [Testing](#testing)
- [Code Style & Standards](#code-style--standards)
- [Debugging](#debugging)
- [Contributing](#contributing)

## Prerequisites

### System Requirements
- **Operating System**: macOS, Linux, or Windows 10/11 with WSL2
- **Memory**: 8GB RAM minimum (16GB recommended)
- **Storage**: 10GB free space
- **Python**: 3.11 or higher
- **Node.js**: 18.0 or higher
- **Git**: Latest version

### Required Software

#### Core Development Tools
```bash
# Python 3.11+
python --version  # Should be 3.11+

# Node.js 18+
node --version    # Should be 18+
npm --version

# Git
git --version
```

#### Optional Tools (Recommended)
- **Docker Desktop**: For containerized development
- **VS Code**: With Python and JavaScript extensions
- **Postman**: For API testing
- **pgAdmin**: For PostgreSQL management

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/your-org/intellipdf.git
cd intellipdf
```

### 2. Environment Setup
```bash
# Copy environment template
cp .env.example .env

# Edit the .env file with your configuration
nano .env  # or use your preferred editor
```

### 3. Run Setup Script
```bash
# Make setup script executable
chmod +x scripts/setup.sh

# Run setup
./scripts/setup.sh
```

### 4. Start Development Servers
```bash
# Start all services with Docker Compose
docker-compose up -d

# OR start services individually (see sections below)
```

### 5. Verify Installation
```bash
# Check backend API
curl http://localhost:8000/health

# Check frontend
curl http://localhost:3000
```

## Project Structure

```
intellipdf/
├── backend/          # FastAPI Python backend
├── frontend/         # Next.js React frontend
├── data/            # Development data storage
├── scripts/         # Utility scripts
├── docs/           # Documentation
└── deployment/     # Deployment configurations
```

### Backend Structure
```
backend/
├── app/
│   ├── api/            # API routes and endpoints
│   ├── core/           # Core business logic
│   ├── schemas/        # Pydantic models
│   └── tests/          # Unit and integration tests
├── requirements.txt    # Python dependencies
└── Dockerfile         # Container configuration
```

### Frontend Structure
```
frontend/
├── src/
│   ├── app/            # Next.js app router pages
│   ├── components/     # React components
│   ├── hooks/          # Custom React hooks
│   ├── lib/            # Utility libraries
│   └── store/          # State management
├── package.json        # Node.js dependencies
└── next.config.js     # Next.js configuration
```

## Development Environment

### Environment Variables

Create a `.env` file in the project root:

```bash
# Development Environment
NODE_ENV=development
DEBUG=true
API_BASE_URL=http://localhost:8000
FRONTEND_URL=http://localhost:3000

# Database
DATABASE_URL=sqlite:///./data/sqlite/intellipdf.db
REDIS_URL=redis://localhost:6379/0

# AI Services (Get these from respective providers)
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=your_pinecone_environment

# File Storage (Local development)
STORAGE_TYPE=local
UPLOAD_DIR=./data/uploads
VECTOR_DB_PATH=./data/vector_db

# Security (Development only - change in production)
JWT_SECRET=dev_jwt_secret_change_in_production
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000

# Performance
MAX_UPLOAD_SIZE=50MB
VECTOR_DIMENSION=384
```

### Docker Development Setup

#### Using Docker Compose (Recommended)
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild services
docker-compose up --build
```

#### Individual Service Setup
```bash
# Backend only
docker-compose up -d db redis
cd backend && docker build -t intellipdf-backend .
docker run -p 8000:8000 --env-file ../.env intellipdf-backend

# Frontend only
cd frontend && docker build -t intellipdf-frontend .
docker run -p 3000:3000 --env-file ../.env intellipdf-frontend
```

### Native Development Setup

#### Backend Setup
```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt

# Run database migrations
alembic upgrade head

# Start development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend Setup
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev

# Alternative: Start with custom port
npm run dev -- --port 3001
```

## Running the Application

### Development Mode

#### Start All Services
```bash
# Using Docker Compose (recommended)
docker-compose up -d

# Using native setup
./scripts/start_dev.sh
```

#### Start Individual Services

**Backend API**:
```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Frontend**:
```bash
cd frontend
npm run dev
```

**Background Workers**:
```bash
cd backend
source venv/bin/activate
celery -A app.main worker --loglevel=info
```

### Service URLs
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **RedisInsight**: http://localhost:8001 (if using Redis Docker)

## API Development

### FastAPI Backend

#### Adding New Endpoints
```python
# backend/app/api/v1/endpoints/example.py
from fastapi import APIRouter, Depends
from app.schemas.example import ExampleResponse
from app.core.ai.example_service import ExampleService

router = APIRouter()

@router.get("/example", response_model=ExampleResponse)
async def get_example(
    service: ExampleService = Depends()
):
    """Get example data."""
    result = await service.get_example()
    return ExampleResponse(**result)
```

#### Database Models
```python
# backend/app/core/database/models.py
from sqlalchemy import Column, Integer, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
```

#### Pydantic Schemas
```python
# backend/app/schemas/document.py
from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class DocumentBase(BaseModel):
    filename: str
    content: Optional[str] = None

class DocumentCreate(DocumentBase):
    pass

class DocumentResponse(DocumentBase):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True
```

### API Testing

#### Unit Tests
```python
# backend/app/tests/test_documents.py
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_upload_document():
    with open("test_document.pdf", "rb") as f:
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": f}
        )
    assert response.status_code == 200
    assert "id" in response.json()
```

#### Manual Testing with curl
```bash
# Health check
curl http://localhost:8000/health

# Upload document
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@sample.pdf"

# Search documents
curl -X POST "http://localhost:8000/api/v1/search" \
     -H "Content-Type: application/json" \
     -d '{"query": "machine learning", "limit": 5}'
```

## Frontend Development

### Next.js Structure

#### Pages and Routing
```typescript
// frontend/src/app/documents/page.tsx
'use client';

import { useState, useEffect } from 'react';
import { DocumentList } from '@/components/document/DocumentList';
import { useDocuments } from '@/hooks/useDocuments';

export default function DocumentsPage() {
  const { documents, loading, error } = useDocuments();

  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error.message}</div>;

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">Documents</h1>
      <DocumentList documents={documents} />
    </div>
  );
}
```

#### Components
```typescript
// frontend/src/components/document/DocumentCard.tsx
import { Document } from '@/lib/types';
import { Card, CardContent, CardHeader } from '@/components/ui/card';

interface DocumentCardProps {
  document: Document;
  onSelect?: (document: Document) => void;
}

export function DocumentCard({ document, onSelect }: DocumentCardProps) {
  return (
    <Card className="cursor-pointer hover:shadow-lg transition-shadow">
      <CardHeader>
        <h3 className="font-semibold">{document.filename}</h3>
      </CardHeader>
      <CardContent>
        <p className="text-sm text-gray-600">
          {document.created_at.toLocaleDateString()}
        </p>
      </CardContent>
    </Card>
  );
}
```

#### Custom Hooks
```typescript
// frontend/src/hooks/useDocuments.ts
import { useState, useEffect } from 'react';
import { apiClient } from '@/lib/api';
import { Document } from '@/lib/types';

export function useDocuments() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    async function fetchDocuments() {
      try {
        const response = await apiClient.get('/documents');
        setDocuments(response.data);
      } catch (err) {
        setError(err as Error);
      } finally {
        setLoading(false);
      }
    }

    fetchDocuments();
  }, []);

  return { documents, loading, error };
}
```

### State Management with Zustand
```typescript
// frontend/src/store/documentStore.ts
import { create } from 'zustand';
import { Document } from '@/lib/types';

interface DocumentState {
  documents: Document[];
  selectedDocument: Document | null;
  setDocuments: (documents: Document[]) => void;
  selectDocument: (document: Document) => void;
  addDocument: (document: Document) => void;
}

export const useDocumentStore = create<DocumentState>((set) => ({
  documents: [],
  selectedDocument: null,
  setDocuments: (documents) => set({ documents }),
  selectDocument: (document) => set({ selectedDocument: document }),
  addDocument: (document) => set((state) => ({
    documents: [...state.documents, document]
  })),
}));
```

### Styling with Tailwind CSS
```typescript
// Example component with Tailwind classes
export function Button({ children, variant = 'primary', ...props }) {
  const baseClasses = 'px-4 py-2 rounded-md font-medium transition-colors';
  const variants = {
    primary: 'bg-blue-600 text-white hover:bg-blue-700',
    secondary: 'bg-gray-200 text-gray-900 hover:bg-gray-300',
  };

  return (
    <button 
      className={`${baseClasses} ${variants[variant]}`}
      {...props}
    >
      {children}
    </button>