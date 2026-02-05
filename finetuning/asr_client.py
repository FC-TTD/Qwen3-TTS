import logging
import os
import sys
from typing import Any, Dict

import requests

logger = logging.getLogger(__name__)


DEFAULT_ASR_HOST = os.environ.get("ASR_HOST") or os.environ.get("ASR_BASE_URL") or "asrpri-api"


def _ensure_base_url(host: str) -> str:
    if host.startswith("http://") or host.startswith("https://"):
        return host.rstrip("/")
    return f"http://{host}".rstrip("/")


def extract_text(response: Dict[str, Any]) -> str:
    try:
        results = response.get("result")
        if isinstance(results, list) and results:
            first = results[0]
            if isinstance(first, dict):
                return str(first.get("clean_text") or first.get("text") or "").strip()
        if isinstance(results, dict):
            return str(results.get("clean_text") or results.get("text") or "").strip()
    except Exception:
        logger.exception("Failed to parse ASR response")
    return ""


def auto_asr(audio_path: str, host: str = DEFAULT_ASR_HOST, language: str = "auto", timeout: int = 120) -> str:
    if not audio_path or os.path.exists(audio_path) is False:
        return ""
    try:
        with open(audio_path, "rb") as audio_data:
            res = asr_model(audio_data, host=host, language=language, timeout=timeout)
        return extract_text(res)
    except Exception:
        logger.exception("ASR failed for %s", audio_path)
        return ""


def asr_model(audio_data, host: str = DEFAULT_ASR_HOST, language: str = "auto", timeout: int = 120) -> Dict[str, Any]:
    """
    curl -X 'POST' \
     'http://asrpri-api/api/v1/asr' \
      -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'files=@.wav;type=audio/wav' \
  -F 'keys=1' \
  -F 'lang=auto'
    """
    headers = {"accept": "application/json"}
    data = {"keys": "wav", "lang": language}
    base_url = _ensure_base_url(host)
    response = requests.post(
        f"{base_url}/api/v1/asr",
        files={"files": audio_data},
        data=data,
        headers=headers,
        timeout=timeout,
    )
    return response.json()


if __name__ == "__main__":
    args = sys.argv[1:]
    audio_data = open(args[0], "rb")
    res = asr_model(audio_data, DEFAULT_ASR_HOST)
    print(extract_text(res))
