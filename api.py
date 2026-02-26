from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from main import Config as RuntimeConfig
from main import initialize_llm, translate_segment


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to translate")
    direction: Literal["lat-en", "en-lat"] = Field(
        "lat-en",
        description="Translation direction: lat-en (Latin to English) or en-lat (English to Latin)",
    )


class TranslateResponse(BaseModel):
    translation: str
    direction: Literal["lat-en", "en-lat"]


class FileTranslateRequest(BaseModel):
    path: str = Field(..., description="Absolute or relative path to a .txt file on the server")
    direction: Literal["lat-en", "en-lat"] = Field(
        "lat-en",
        description="Translation direction: lat-en (Latin to English) or en-lat (English to Latin)",
    )


class HealthResponse(BaseModel):
    status: str
    backend: str
    model: str
    llm_initialized: bool


app = FastAPI(
    title="Latin RAG Translator API",
    version="0.2.0",
    description=(
        "Bidirectional Latin<->English translation API. "
        "Current backend uses Ollama via `main.py` configuration."
    ),
)


class TranslatorService:
    def __init__(self) -> None:
        self.llm: Optional[object] = None

    def _ensure_llm(self):
        if self.llm is None:
            self.llm = initialize_llm()
        return self.llm

    def translate_text(self, text: str, direction: str) -> str:
        llm = self._ensure_llm()
        return translate_segment(llm, text, direction)

    def translate_file(self, input_file: Path, direction: str) -> str:
        llm = self._ensure_llm()

        if not input_file.exists():
            raise FileNotFoundError(f"File not found: {input_file}")
        if input_file.suffix.lower() != ".txt":
            raise ValueError("Only .txt files are supported for now")

        lines = input_file.read_text(encoding="utf-8").splitlines()
        outputs = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            handled = False

            # Book title (all caps, starts with LIBER)
            if line.isupper() and line.startswith("LIBER"):
                source_lang = "Latin" if direction == "lat-en" else "English"
                target_lang = "English" if direction == "lat-en" else "Latin"
                result = llm(
                    f"Translate this {source_lang} book title to {target_lang}: {line}"
                ).strip()
                outputs.append(result)
                handled = True

            # Section title (starts with capital, not digit)
            if not handled and line[0].isupper() and not line[0].isdigit():
                source_lang = "Latin" if direction == "lat-en" else "English"
                target_lang = "English" if direction == "lat-en" else "Latin"
                result = llm(
                    f"Translate this {source_lang} section title to {target_lang}: {line}"
                ).strip()
                outputs.append(result)
                handled = True

            # Numbered section (starts with number)
            if not handled and line[0].isdigit():
                period_positions = [pos for pos, char in enumerate(line) if char == "."]
                if len(period_positions) >= 2:
                    section_nums = line[: period_positions[1] + 1].strip()
                    content = line[period_positions[1] + 1 :].strip()

                    source_lang = "Latin" if direction == "lat-en" else "English"
                    target_lang = "English" if direction == "lat-en" else "Latin"
                    result = llm(
                        f"Translate this {source_lang} text to {target_lang}: {content}"
                    ).strip()

                    outputs.append(f"{section_nums}: {result}")
                    handled = True

            # Fallback: translate the whole line
            if not handled:
                outputs.append(translate_segment(llm, line, direction))

        return "\n\n".join(outputs).strip()


translator_service = TranslatorService()


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """Basic health check without forcing LLM startup."""

    return HealthResponse(
        status="ok",
        backend="ollama",
        model=RuntimeConfig.LLM_MODEL,
        llm_initialized=translator_service.llm is not None,
    )


@app.get("/model-info", response_model=HealthResponse, tags=["system"])
def model_info() -> HealthResponse:
    """Return active backend/model info and whether the LLM client is initialized."""

    return health()


@app.post("/translate", response_model=TranslateResponse, tags=["translation"])
def translate(request: TranslateRequest) -> TranslateResponse:
    """Translate a single text fragment in either direction."""

    try:
        translation = translator_service.translate_text(request.text, request.direction)
    except Exception as exc:  # pragma: no cover - FastAPI handles status codes
        raise HTTPException(status_code=500, detail=str(exc))
    return TranslateResponse(translation=translation, direction=request.direction)


@app.post("/translate-file", response_model=TranslateResponse, tags=["translation"])
def translate_file(request: FileTranslateRequest) -> TranslateResponse:
    """Translate a UTF-8 `.txt` file line by line on the server."""

    try:
        translation = translator_service.translate_file(Path(request.path), request.direction)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # pragma: no cover - FastAPI handles status codes
        raise HTTPException(status_code=500, detail=str(exc))

    return TranslateResponse(translation=translation, direction=request.direction)
