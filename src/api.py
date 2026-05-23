import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.generation.main import initialize_search_pipeline, run_query


# Global state to hold the initialized pipeline
pipeline = None


class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    question: str = Field(..., description="The question to ask the RAG system")


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    answer: str = Field(..., description="The generated answer")
    is_cached: bool = Field(..., description="Whether the response was served from cache")
    tokens_used: int = Field(..., description="Number of tokens used in generation")
    time_taken: float = Field(..., description="Time taken to process the query in seconds")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to load/unload the search pipeline."""
    global pipeline

    # Startup: Initialize the pipeline
    print("Initializing search pipeline...")
    pipeline = initialize_search_pipeline()
    print("Search pipeline initialized successfully!")

    yield  # Application runs here

    # Shutdown: Cleanup if needed
    print("Shutting down...")
    pipeline = None


app = FastAPI(
    title="Enterprise RAG API",
    description="Simple API for querying the Enterprise RAG system",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html() -> HTMLResponse:
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - Swagger UI",
        swagger_ui_parameters={
            "syntaxHighlight.theme": "obsidian"
        }
    )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to Enterprise RAG API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "query": "/query (POST)"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if pipeline is not None else "unhealthy",
        "pipeline_loaded": pipeline is not None
    }


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Query the RAG system with a question.

    Returns the answer along with metadata about caching, tokens used, and processing time.
    """
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search pipeline not initialized. Please wait for startup or check server status."
        )

    if not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty"
        )

    try:
        start_time = time.perf_counter()

        # Unpack pipeline components
        (pc, pc_index, dense_tokenizer, dense_model,
         sparse_tokenizer, sparse_model, sparse_input_names,
         reranker) = pipeline

        # Run the query
        (is_cache_hit, retrieved_docs, answer,
         tokens_used, finish_reason) = run_query(
            query=request.question,
            pc=pc, pc_index=pc_index,
            dense_tokenizer=dense_tokenizer, dense_model=dense_model,
            sparse_tokenizer=sparse_tokenizer, sparse_model=sparse_model,
            sparse_input_names=sparse_input_names,
            reranker_model=reranker
        )

        end_time = time.perf_counter()
        time_taken = end_time - start_time

        return QueryResponse(
            answer=answer,
            is_cached=is_cache_hit,
            tokens_used=tokens_used,
            time_taken=round(time_taken, 2)
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)