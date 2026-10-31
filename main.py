import os
import httpx

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="CodeForge AI",
    description="AI-powered software engineering platform",
    version="0.1.0"
)

# Frontend access. We will restrict this to our actual website later.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=12000)
    language: str = Field(default="python", max_length=50)


@app.get("/")
async def home():
    return {
        "name": "CodeForge AI",
        "status": "online",
        "version": "0.1.0",
        "message": "Your AI software engineering platform"
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/api/generate")
async def generate_code(request: GenerateRequest):

    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("OPENROUTER_MODEL", "")

    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="OpenRouter API key has not been configured."
        )

    # Strictly prevent accidental use of paid models.
    if not model.endswith(":free"):
        raise HTTPException(
            status_code=503,
            detail="Free models only. Configure an available :free model."
        )

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are CodeForge AI, an expert software engineering "
                    "assistant. Generate clear, maintainable and secure code. "
                    "Explain important implementation decisions. "
                    "Never claim that code is error-free without testing."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Programming language: {request.language}\n\n"
                    f"Task:\n{request.prompt}"
                )
            }
        ]
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://codeforge-ai.onrender.com",
        "X-Title": "CodeForge AI"
    }

    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload
            )

        if response.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail=(
                    "The AI provider could not complete the request. "
                    "Check model availability and free-tier limits."
                )
            )

        data = response.json()

        return {
            "status": "success",
            "model": model,
            "language": request.language,
            "generated_code": data["choices"][0]["message"]["content"]
        }

    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="The AI request timed out. Please try again."
        )

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="An unexpected server error occurred."
        )
