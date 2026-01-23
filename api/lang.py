from __future__ import annotations

from typing import Callable, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse


ModelGetter = Callable[[], object]


def normalize_language(lang: Optional[str]) -> str:
    v = (lang or "auto").strip().lower()
    return v or "auto"


def get_supported_languages(model_getter: ModelGetter) -> Optional[List[str]]:
    model = model_getter()
    if model is None:
        return None

    langs = getattr(model, "get_supported_languages", None)
    if not callable(langs):
        return None

    v = langs()
    if v is None:
        return None

    return [str(x).strip().lower() for x in v if x is not None and str(x).strip()]


def validate_language_or_400(language: Optional[str], model_getter: ModelGetter) -> str:
    language_norm = normalize_language(language)
    supported_langs = get_supported_languages(model_getter)

    if supported_langs is not None and language_norm not in set(supported_langs):
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"Unsupported language: {language}",
                "supported": sorted(set(supported_langs)),
            },
        )

    return language_norm


def setup_language_openapi(app, model_getter: ModelGetter) -> None:
    orig_openapi = app.openapi

    def _custom_openapi():
        schema = orig_openapi()
        langs = get_supported_languages(model_getter)
        if not langs:
            return schema

        try:
            props = (
                schema["paths"]["/api/tts"]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"][
                    "properties"
                ]
            )
            if isinstance(props, dict) and "language" in props and isinstance(props["language"], dict):
                props["language"]["enum"] = sorted(set(langs))
                props["language"].setdefault("default", "auto")
        except Exception:
            return schema

        return schema

    app.openapi = _custom_openapi


def create_router(model_getter: ModelGetter) -> APIRouter:
    router = APIRouter()

    @router.get("/languages")
    async def get_languages():
        model = model_getter()
        if model is None:
            raise HTTPException(status_code=503, detail="Model not initialized")

        langs = get_supported_languages(model_getter)
        return JSONResponse(content={"languages": langs})

    return router
