# backend/main.py

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .rag import answer_question, build_or_load_index

app = FastAPI(
    title="Chatbot RAG Consular",
    description="API para consultas de trámites consulares usando RAG sobre knowledge.txt",
)

# --- CORS (por si luego lo usas desde otro dominio, p.ej. GitHub Pages) ---
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https://.*\.app\.github\.dev",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str


@app.on_event("startup")
async def startup_event():
    """
    Al iniciar el servidor, intentamos cargar (o construir) el índice.
    Así el primer request es más rápido.
    """
    try:
        build_or_load_index()
        print("[API] Índice RAG cargado correctamente.")
    except Exception as e:
        print(f"[API] Error al construir/cargar índice: {e}")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AnswerResponse)
async def ask(req: QuestionRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía.")

    try:
        ans = answer_question(req.question)
        return AnswerResponse(answer=ans)
    except Exception as e:
        # Log simple
        print(f"[API] Error procesando pregunta: {e}")
        raise HTTPException(
            status_code=500,
            detail="Ocurrió un error al procesar la consulta en el asistente.",
        )


# --- Servir el frontend estático desde FastAPI (mismo puerto 8001) ---

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"

# Esto hace que / sirva index.html de frontend, y /css, /js, /assets, etc.
app.mount(
    "/",
    StaticFiles(directory=str(FRONTEND_DIR), html=True),
    name="frontend",
)
