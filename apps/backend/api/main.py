"""
FastAPI Ana Uygulaması
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from api.endpoints import router
import uvicorn


def create_app() -> FastAPI:
    """
    FastAPI uygulamasını oluşturur ve yapılandırır.

    Returns:
        FastAPI uygulaması
    """
    app = FastAPI(
        title="AI Human Detector API",
        description="Yapay zeka tarafından üretilen insan görsellerini tespit eden API",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Production'da spesifik origin'ler kullanın
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Router'ı ekle
    app.include_router(router)

    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "name": "AI Human Detector API",
            "version": "0.1.0",
            "status": "running",
            "docs": "/docs",
            "endpoints": {
                "health": "/health",
                "analyze": "/api/v1/analyze",
                "models": "/models"
            }
        }

    # Health check endpoint (alternate)
    @app.get("/ping")
    async def ping():
        return {"status": "pong"}

    return app


# Uygulama örneği
app = create_app()


def main():
    """
    Development server'ı başlatır.
    """
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()
