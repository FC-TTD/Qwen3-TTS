from __future__ import annotations

from io import BytesIO
from typing import Optional

import httpx
from fastapi import HTTPException, UploadFile
from fastapi.responses import Response


class BaseProxy:
    def __init__(self, base_url: str, timeout_seconds: float):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds

    async def synthesize(
        self,
        *,
        text: str,
        language: str,
        ref_audio: Optional[UploadFile],
        ref_text: Optional[str],
        x_vector_only_mode: bool,
        remove_silence: bool,
        speed: float,
        expected_duration: Optional[float],
        postprocess: bool,
        lufs: float,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        max_new_tokens: int,
    ) -> Response:
        if ref_audio is None:
            raise HTTPException(status_code=400, detail="voice_clone mode requires ref_audio")

        audio_bytes = await ref_audio.read()
        files = {
            "ref_audio": (ref_audio.filename or "reference.wav", BytesIO(audio_bytes), ref_audio.content_type or "audio/wav"),
        }
        data = {
            "text": text,
            "language": language,
            "mode": "voice_clone",
            "x_vector_only_mode": str(x_vector_only_mode).lower(),
            "remove_silence": str(remove_silence).lower(),
            "speed": str(speed),
            "postprocess": str(postprocess).lower(),
            "lufs": str(lufs),
            "temperature": str(temperature),
            "top_p": str(top_p),
            "top_k": str(top_k),
            "repetition_penalty": str(repetition_penalty),
            "max_new_tokens": str(max_new_tokens),
        }
        if ref_text is not None:
            data["ref_text"] = ref_text
        if expected_duration is not None:
            data["expected_duration"] = str(expected_duration)

        timeout = httpx.Timeout(self._timeout, connect=min(10.0, self._timeout))
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                resp = await client.post(f"{self._base_url}/api/tts", data=data, files=files)
            except httpx.HTTPError as exc:
                raise HTTPException(status_code=502, detail=f"base upstream unavailable: {exc}") from exc

        if resp.status_code >= 400:
            detail = resp.text
            try:
                detail = resp.json()
            except ValueError:
                pass
            raise HTTPException(status_code=resp.status_code, detail=detail)

        return Response(content=resp.content, media_type=resp.headers.get("content-type", "audio/wav"))

    def synthesize_sync(
        self,
        *,
        text: str,
        language: str,
        audio_bytes: bytes,
        filename: str,
        content_type: str,
        ref_text: Optional[str],
        x_vector_only_mode: bool,
        remove_silence: bool,
        speed: float,
        expected_duration: Optional[float],
        postprocess: bool,
        lufs: float,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        max_new_tokens: int,
    ) -> bytes:
        files = {
            "ref_audio": (filename, BytesIO(audio_bytes), content_type or "audio/wav"),
        }
        data = {
            "text": text,
            "language": language,
            "mode": "voice_clone",
            "x_vector_only_mode": str(x_vector_only_mode).lower(),
            "remove_silence": str(remove_silence).lower(),
            "speed": str(speed),
            "postprocess": str(postprocess).lower(),
            "lufs": str(lufs),
            "temperature": str(temperature),
            "top_p": str(top_p),
            "top_k": str(top_k),
            "repetition_penalty": str(repetition_penalty),
            "max_new_tokens": str(max_new_tokens),
        }
        if ref_text is not None:
            data["ref_text"] = ref_text
        if expected_duration is not None:
            data["expected_duration"] = str(expected_duration)

        timeout = httpx.Timeout(self._timeout, connect=min(10.0, self._timeout))
        with httpx.Client(timeout=timeout) as client:
            try:
                resp = client.post(f"{self._base_url}/api/tts", data=data, files=files)
            except httpx.HTTPError as exc:
                raise RuntimeError(f"base upstream unavailable: {exc}") from exc

        if resp.status_code >= 400:
            detail = resp.text
            try:
                detail = resp.json()
            except ValueError:
                pass
            raise RuntimeError(f"base upstream error {resp.status_code}: {detail}")

        return resp.content

    async def health(self) -> dict:
        timeout = httpx.Timeout(10.0, connect=5.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                resp = await client.get(f"{self._base_url}/health")
                return {"ok": resp.status_code == 200, "status_code": resp.status_code}
            except httpx.HTTPError as exc:
                return {"ok": False, "error": str(exc)}
