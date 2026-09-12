#!/usr/bin/env python3
"""Real GPU/HTTP acceptance; writes generated WAV evidence only to --output."""
import argparse
from io import BytesIO
import json
from pathlib import Path
import time

import numpy as np
import requests
import soundfile as sf

TEXT = "Hello. This is a short speech synthesis test. The three voices share one model service."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://qwen-fusion")
    parser.add_argument("--output", required=True)
    parser.add_argument("--gradio", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    records = []

    def generate(name, mode, **overrides):
        data = dict(text=TEXT, language="English", mode=mode, postprocess="true",
                    top_k=1, temperature=0.1, max_new_tokens=1024, **overrides)
        files = None
        if mode == "voice_clone":
            data["ref_text"] = TEXT
            files = {"ref_audio": ("reference.wav", (output / "design.wav").read_bytes(), "audio/wav")}
        started = time.monotonic()
        response = session.post(args.url + "/api/tts", data=data, files=files, timeout=900)
        response.raise_for_status()
        wav, sr = sf.read(BytesIO(response.content))
        assert sr == 48000 and wav.size > 0 and np.isfinite(wav).all()
        assert np.max(abs(wav)) > 1e-4, "silent output"
        (output / f"{name}.wav").write_bytes(response.content)
        record = dict(name=name, mode=mode, sample_rate=sr, duration=len(wav)/sr,
                      elapsed=time.monotonic()-started, bytes=len(response.content))
        records.append(record)
        print(json.dumps(record), flush=True)
        return record

    for route in ["/health", "/health/backends", "/languages", "/api/speakers"]:
        response = session.get(args.url + route, timeout=10)
        response.raise_for_status()
        print(route, response.text[:500])
    generate("design", "voice_design", instruct="A warm, clear adult English voice.")
    generate("custom", "custom_voice", speaker="Ryan")
    normal = generate("clone", "voice_clone")
    slow = generate("clone-slow", "voice_clone", speed=0.65)
    fast = generate("clone-fast", "voice_clone", speed=1.5)
    target = normal["duration"] * 0.85
    aligned = generate("clone-aligned", "voice_clone", expected_duration=target)
    # Sampling can vary duration; save all measurements for review as well.
    assert slow["duration"] > fast["duration"], "real slow/fast ordering failed"
    assert abs(aligned["duration"]-target)/target < 0.15, "duration alignment outside 15% smoke tolerance"
    for key, value in [("speed", 0), ("expected_duration", -1)]:
        response = session.post(args.url + "/api/tts", data={"text": "hello", "mode": "voice_design", "instruct": "Warm", key: value}, timeout=10)
        assert response.status_code == 400, (key, response.status_code, response.text)
    if args.gradio:
        from gradio_client import Client, handle_file
        client = Client(args.url)
        results = {
            "design": client.predict(TEXT, "English", "Warm adult English voice", api_name="/generate_design"),
            "custom": client.predict(TEXT, "English", "Ryan", "", "1.7B", "default", api_name="/generate_custom"),
            "clone": client.predict(handle_file(str(output / "design.wav")), TEXT, TEXT, "English", False, api_name="/generate_clone"),
        }
        for name, result in results.items():
            assert result[0], (name, result)
            wav, sr = sf.read(result[0])
            assert sr == 48000 and len(wav) > 0 and np.isfinite(wav).all()
            sf.write(output / f"ui-{name}.wav", wav, sr)
    response = session.post(args.url + "/api/unload", timeout=900)
    response.raise_for_status()
    assert response.json()["loaded_model"] is None
    (output / "evidence.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
