from fastapi import FastAPI, HTTPException, Depends, status, Security, Request, Response
from fastapi.security import APIKeyHeader
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from neo4j import GraphDatabase
import ollama
import os
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load configuration from environment variables
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://10.171.63.13:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "lzz928666")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "my-custom-model:latest")
API_KEY = os.getenv("API_KEY", "lzz928666")  # Change this in production

# API Key authentication setup
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key"
        )
    return api_key

app = FastAPI(
    title="Knowledge Graph QA API",
    description="API for querying and asking questions about knowledge graph",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Root endpoint
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>Knowledge Graph QA API</title>
            <link rel="icon" href="data:,"> <!-- Disable favicon -->
        </head>
        <body>
            <h1>Knowledge Graph QA API</h1>
            <p>Visit <a href="/docs">/docs</a> for API documentation</p>
            <p>Include your API key in headers: <code>X-API-Key: your-secret-key-here</code></p>
        </body>
    </html>
    """

# Middleware to suppress favicon 404 logs
@app.middleware("http")
async def suppress_favicon(request: Request, call_next):
    if request.url.path == "/favicon.ico":
        return Response(status_code=204)
    return await call_next(request)

class KnowledgeGraphQA:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI, 
            auth=(NEO4J_USER, NEO4J_PASSWORD),
            max_connection_lifetime=30*60  # 30 minutes
        )
        self.node_indexes = [
            "act_name_ft", "appcon_name_ft", "fun_name_ft",
            "iden_name_ft", "strcom_name_ft", "value_name_ft"
        ]
        self.model_name = OLLAMA_MODEL

    def close(self):
        self.driver.close()

    def query_neo4j(self, query: str) -> List[Any]:
        try:
            with self.driver.session() as session:
                result = session.run(query)
                return [record for record in result]
        except Exception as e:
            logger.error(f"Neo4j query failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database connection error"
            )

    def get_related_nodes(self, keyword: str) -> List[Dict[str, Any]]:
        all_results = []
        for index_name in self.node_indexes:
            cypher = f"""
            CALL db.index.fulltext.queryNodes("{index_name}", "{keyword}")
            YIELD node, score
            MATCH (node)-[r]-(related)
            RETURN DISTINCT
                node,
                type(r) AS relationship_type,
                related,
                score,
                labels(node) AS node_labels,
                labels(related) AS related_labels
            ORDER BY score DESC
            LIMIT 5
            """
            try:
                results = self.query_neo4j(cypher)
                all_results.extend([self._record_to_dict(record) for record in results])
            except Exception as e:
                logger.warning(f"Query on index {index_name} failed: {str(e)}")
                continue
        return all_results

    @staticmethod
    def _record_to_dict(record) -> Dict[str, Any]:
        """Convert Neo4j record to serializable dict"""
        return {
            "node": dict(record["node"]),
            "relationship_type": record["relationship_type"],
            "related": dict(record["related"]),
            "score": float(record["score"]),
            "node_labels": list(record["node_labels"]),
            "related_labels": list(record["related_labels"])
        }

    @staticmethod
    def format_props(props: Dict[str, Any]) -> str:
        excluded = ['name', 'NAME', 'id', 'ID']
        items = [f"{k}: {v}" for k, v in props.items() if k not in excluded]
        return ", ".join(items) if items else "no other properties"

    def format_context(self, results: List[Dict[str, Any]]) -> str:
        context_parts = []
        for record in results:
            node = record['node']
            rel_type = record['relationship_type']
            related = record['related']
            node_labels = record['node_labels']
            related_labels = record['related_labels']

            node_props = {k: v for k, v in node.items() 
                         if v is not None and str(v).lower() not in ('nan', 'none', '')}
            related_props = {k: v for k, v in related.items() 
                            if v is not None and str(v).lower() not in ('nan', 'none', '')}

            node_name = node_props.get('name', 'Unknown Entity')
            related_name = related_props.get('name', 'Unknown Entity')

            part = (
                f"{'/'.join(node_labels)} entity '{node_name}' "
                f"(properties: {self.format_props(node_props)})\n"
                f"is connected via '{rel_type}' relationship to\n"
                f"{'/'.join(related_labels)} entity '{related_name}' "
                f"(properties: {self.format_props(related_props)})"
            )
            context_parts.append(part)
        return "\n\n".join(context_parts)

    def generate_response(self, question: str, context: str) -> str:
        prompt = f"""
You are an expert in knowledge graph analysis. Please provide a professional answer based on the following:

Question:
{question}

Related knowledge graph context:
{context}

Answer requirements:
1. First, answer the question directly.
2. Then, provide a detailed explanation combined with the knowledge graph information.
3. Use clear and professional English.
"""
        try:
            response = ollama.generate(model=self.model_name, prompt=prompt)
            return response['response']
        except Exception as e:
            logger.error(f"Ollama generation failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="LLM service unavailable"
            )

# Initialize service
kgqa = KnowledgeGraphQA()

# API request models
class SearchRequest(BaseModel):
    keyword: str
    max_results: Optional[int] = 5

class QuestionRequest(BaseModel):
    question: str
    context: Optional[str] = None
    raw_data: Optional[List[Dict[str, Any]]] = None

class HealthCheckResponse(BaseModel):
    status: str
    neo4j_connected: bool
    ollama_available: bool
    timestamp: datetime

# API endpoints
@app.get("/health", response_model=HealthCheckResponse)
async def health_check(api_key: str = Depends(get_api_key)):
    """Service health check"""
    neo4j_status = False
    ollama_status = False
    
    try:
        with kgqa.driver.session() as session:
            session.run("RETURN 1")
            neo4j_status = True
    except:
        pass
    
    try:
        ollama.list()  # Simple Ollama availability check
        ollama_status = True
    except:
        pass
    
    return {
        "status": "Service healthy",
        "neo4j_connected": neo4j_status,
        "ollama_available": ollama_status,
        "timestamp": datetime.now()
    }

@app.post("/search", response_model=Dict[str, Any])
async def search_knowledge_graph(
    request: SearchRequest, 
    api_key: str = Depends(get_api_key)
):
    """
    Search knowledge graph
    
    - **keyword**: Search term
    - **max_results**: Max results per index (default 5)
    """
    try:
        start_time = datetime.now()
        logger.info(f"Searching for keyword: {request.keyword}")
        
        results = kgqa.get_related_nodes(request.keyword)
        formatted_context = kgqa.format_context(results)
        
        return {
            "status": "success",
            "keyword": request.keyword,
            "results_count": len(results),
            "context": formatted_context,
            "raw_data": results[:request.max_results * len(kgqa.node_indexes)],
            "processing_time": str(datetime.now() - start_time)
        }
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )

@app.post("/ask", response_model=Dict[str, Any])
async def ask_question(
    request: QuestionRequest, 
    api_key: str = Depends(get_api_key)
):
    """
    Answer questions based on knowledge graph
    
    - **question**: Question to answer
    - **context**: Optional context (if not provided, will use raw_data)
    - **raw_data**: Optional raw knowledge graph data
    """
    try:
        start_time = datetime.now()
        
        if not request.question:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Question cannot be empty"
            )
        
        # Determine which context to use
        used_context = request.context
        if not used_context and request.raw_data:
            used_context = kgqa.format_context(request.raw_data)
        
        if not used_context:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either context or raw_data must be provided"
            )
        
        logger.info(f"Generating answer for question: {request.question[:50]}...")
        answer = kgqa.generate_response(request.question, used_context)
        
        return {
            "status": "success",
            "question": request.question,
            "answer": answer,
            "context_used": used_context[:200] + "..." if len(used_context) > 200 else used_context,
            "processing_time": str(datetime.now() - start_time)
        }
    except Exception as e:
        logger.error(f"Question answering failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Question answering failed: {str(e)}"
        )

# Application lifecycle events
@app.on_event("startup")
async def startup_event():
    logger.info("Starting Knowledge Graph QA API service...")
    # Test connections
    try:
        with kgqa.driver.session() as session:
            session.run("RETURN 1")
        logger.info("Neo4j connection established")
    except Exception as e:
        logger.error(f"Neo4j connection failed: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Knowledge Graph QA API service...")
    kgqa.close()
    logger.info("Neo4j connection closed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "kgqa_api:app",
        host="0.0.0.0",
        port=8000,  
        reload=True,
        log_level="info"
    )