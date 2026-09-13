"""CPU contract tests; model outputs are fakes, audio processing and ASGI are real."""
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import threading
import time
import weakref

import numpy as np
import pytest
import soundfile as sf
from fastapi import HTTPException
from fastapi.testclient import TestClient

from fusion.audio import encode_wav
from fusion.runtime import FusionRuntime


class FakeModel:
    def __init__(self, key, calls, gate=None):
        self.key, self.calls, self.gate = key, calls, gate

    def get_supported_languages(self):
        return ["auto", "english", "chinese", "italian"]

    def _generate(self, mode, **kwargs):
        self.calls.append((mode, self.key, kwargs))
        if self.gate:
            self.gate[0].set()
            assert self.gate[1].wait(5)
        wave = 0.2 * np.sin(2 * np.pi * 220 * np.arange(48000) / 24000)
        return [wave.astype(np.float32)], 24000

    def generate_voice_clone(self, **kwargs):
        return self._generate("voice_clone", **kwargs)

    def generate_voice_design(self, **kwargs):
        return self._generate("voice_design", **kwargs)

    def generate_custom_voice(self, **kwargs):
        return self._generate("custom_voice", **kwargs)


@pytest.fixture
def runtime():
    calls, loads = [], []
    def loader(key):
        loads.append(key)
        return FakeModel(key, calls)
    rt = FusionRuntime(loader=loader)
    rt.calls, rt.loads = calls, loads
    rt.start()
    yield rt
    rt.shutdown()


def clone_args(**overrides):
    return dict(text="Hello world.", language="English", mode="voice_clone",
                ref_audio=(np.zeros(24000, dtype=np.float32), 24000),
                ref_text="Hello.", postprocess=False, **overrides)


def test_three_modes_share_one_lazy_model_and_forward_parameters(runtime):
    runtime.synthesize(**clone_args(temperature=0.7, max_new_tokens=42))
    runtime.synthesize(text="hello", mode="voice_design", instruct="Warm voice", postprocess=False)
    runtime.synthesize(text="hello", mode="custom_voice", speaker="Ryan", postprocess=False)
    runtime.synthesize(**clone_args())
    assert [x[0] for x in runtime.loads] == ["Base", "VoiceDesign", "CustomVoice", "Base"]
    assert runtime.calls[0][2]["temperature"] == 0.7
    assert runtime.calls[0][2]["max_new_tokens"] == 42
    assert runtime.calls[2][2]["speaker"] == "ryan"
    assert runtime.status()["pending"] == 0
    assert runtime.unload()["loaded_model"] is None


@pytest.mark.parametrize("changes", [
    {"text": " "}, {"speed": 0}, {"speed": float("nan")},
    {"expected_duration": -1}, {"expected_duration": float("inf")},
    {"language": "invalid"}, {"mode": "bad"}, {"ref_audio": None},
    {"ref_text": ""}, {"model_size": "7B"}, {"top_p": 2},
])
def test_invalid_requests_do_not_load_models(runtime, changes):
    args = clone_args()
    args.update(changes)
    with pytest.raises(ValueError):
        runtime.synthesize(**args)
    assert runtime.loads == []


def test_real_audio_speed_and_expected_duration(runtime):
    normal = runtime.synthesize(**clone_args())
    slow = runtime.synthesize(**clone_args(speed=0.8))
    fast = runtime.synthesize(**clone_args(speed=1.2))
    aligned = runtime.synthesize(**clone_args(expected_duration=1.6))
    assert normal[0] == slow[0] == fast[0] == aligned[0] == 48000
    assert len(slow[1]) > len(normal[1]) > len(fast[1])
    assert abs(len(aligned[1]) / 48000 - 1.6) < 0.08
    decoded, sr = sf.read(BytesIO(encode_wav(normal)))
    assert sr == 48000 and np.isfinite(decoded).all() and np.max(abs(decoded)) > 0.1


def test_real_postprocess_and_silence_contract(runtime):
    args = clone_args()
    args.update(postprocess=True, remove_silence=True, lufs=-23.0)
    sr, wav = runtime.synthesize(**args)
    assert sr == 48000 and wav.size > 0 and np.isfinite(wav).all()


def test_switch_and_manual_unload_wait_for_active_inference():
    entered, release = threading.Event(), threading.Event()
    calls, weak_models = [], []
    def loader(key):
        if weak_models:
            assert weak_models[-1]() is None, "previous model still retained during switch"
        model = FakeModel(key, calls, (entered, release) if key[0] == "Base" else None)
        weak_models.append(weakref.ref(model))
        return model
    rt = FusionRuntime(loader=loader, idle_seconds=0)
    rt.start()
    try:
        with ThreadPoolExecutor(2) as pool:
            active = pool.submit(rt.synthesize, **clone_args())
            assert entered.wait(2)
            switched = pool.submit(rt.synthesize, text="hello", mode="voice_design", instruct="Warm", postprocess=False)
            rt.unload_if_idle()
            assert rt.status()["loaded_model"][0] == "Base"
            assert len(weak_models) == 1
            release.set()
            active.result(5)
            switched.result(5)
        rt.unload_if_idle()
        assert weak_models[-1]() is None
    finally:
        release.set()
        rt.shutdown()


def test_queue_overload_and_timeout_leave_runtime_usable():
    entered, release = threading.Event(), threading.Event()
    rt = FusionRuntime(loader=lambda key: FakeModel(key, [], (entered, release)),
                       max_pending=1, queue_timeout=0.05)
    rt.start()
    try:
        with ThreadPoolExecutor(2) as pool:
            active = pool.submit(rt.synthesize, **clone_args())
            assert entered.wait(2)
            waiting = pool.submit(rt.synthesize, **clone_args())
            deadline = time.monotonic() + 2
            while rt.status()["pending"] < 2 and time.monotonic() < deadline:
                time.sleep(0.001)
            with pytest.raises(HTTPException) as full:
                rt.synthesize(**clone_args())
            assert full.value.status_code == 429
            with pytest.raises(HTTPException) as expired:
                waiting.result(2)
            assert expired.value.status_code == 503
            release.set()
            active.result(5)
        assert rt.status()["pending"] == 0
        rt.synthesize(**clone_args())
    finally:
        release.set()
        rt.shutdown()


def test_model_failure_releases_queue_and_can_retry(runtime):
    original_loader = runtime.loader
    def failed_loader(key):
        raise RuntimeError("load failed")
    runtime.loader = failed_loader
    with pytest.raises(RuntimeError):
        runtime.synthesize(**clone_args())
    assert runtime.status()["pending"] == 0
    runtime.loader = original_loader
    runtime.synthesize(**clone_args())


@pytest.fixture
def client(runtime, monkeypatch):
    import fusion.host as host
    monkeypatch.setattr(host, "runtime", runtime)
    with TestClient(host.app) as value:
        yield value


def reference_file():
    return {"ref_audio": ("reference.wav", encode_wav((24000, np.zeros(24000))), "audio/wav")}


def test_http_clone_and_mode_contracts(client, runtime):
    response = client.post("/api/tts", data={"text": "hello", "ref_text": "hello", "postprocess": "false"}, files=reference_file())
    assert response.status_code == 200, response.text
    audio, sr = sf.read(BytesIO(response.content))
    assert sr == 48000 and len(audio) == 96000
    for data in [dict(mode="voice_design", instruct="Warm"), dict(mode="custom_voice", speaker="Ryan")]:
        response = client.post("/api/tts", data={"text": "hello", "postprocess": "false", **data})
        assert response.status_code == 200, response.text
    assert client.get("/languages").json()["languages"] == __import__("fusion.runtime", fromlist=["LANGUAGES"]).LANGUAGES
    assert client.get("/api/speakers").status_code == 200
    assert client.post("/api/unload").json()["loaded_model"] is None
    assert client.get("/").status_code == 200


def test_http_validation(client, runtime):
    for data in [{"text": "hello"}, {"text": "hello", "mode": "nope"},
                 {"text": "hello", "mode": "voice_design", "instruct": "Warm", "speed": "0"}]:
        assert client.post("/api/tts", data=data).status_code == 400
    assert client.post("/api/tts", data={"text": "hello"}, files={"ref_audio": ("bad.wav", b"invalid")}).status_code == 400
    assert client.post("/api/tts", data={"text": "hello", "mode": "voice_design", "instruct": "warm", "speed": "x"}).status_code == 422
    assert runtime.loads == []


def test_http_readiness_stays_responsive_during_generation(client, runtime):
    entered, release = threading.Event(), threading.Event()
    runtime.loader = lambda key: FakeModel(key, [], (entered, release))
    with ThreadPoolExecutor(1) as pool:
        active = pool.submit(client.post, "/api/tts", data={"text": "hello", "ref_text": "hello", "postprocess": "false"}, files=reference_file())
        try:
            assert entered.wait(3)
            started = time.monotonic()
            status = client.get("/health/backends")
            assert status.status_code == 200
            assert status.json()["local"]["busy"]
            assert time.monotonic() - started < 1
        finally:
            release.set()
        assert active.result(5).status_code == 200


def test_gradio_callbacks_use_same_runtime_and_injected_design_callback(runtime):
    from fusion.webui import build_callbacks, build_ui
    from fusion.config import settings
    clone, design, custom = build_callbacks(runtime)
    audio_data = (24000, np.zeros(24000, dtype=np.float32))
    assert clone(audio_data, "hello", "hello", "English", False)[0][0] == 48000
    assert design("hello", "English", "warm")[0][0] == 48000
    assert custom("hello", "English", "Ryan", None, "1.7B", "default")[0][0] == 48000
    ui = build_ui(runtime, settings)
    functions = [v.fn for v in ui.fns.values()]
    names = {f.__name__ for f in functions}
    assert {"generate_clone", "generate_design", "generate_custom"} <= names
    assert "generate_voice_design" not in names


def test_custom_voice_keeps_space_normalization_and_empty_default_checkpoint(runtime):
    runtime.synthesize(text="hello", mode="custom_voice", speaker="Uncle Fu",
                       checkpoint_path="", postprocess=False)
    assert runtime.calls[0][2]["speaker"] == "uncle_fu"
    assert runtime.loads[0] == ("CustomVoice", "1.7B", None)


@pytest.mark.parametrize("phase", ["load", "infer"])
@pytest.mark.parametrize("error_type", [RuntimeError, ValueError, HTTPException])
def test_retained_chained_failure_does_not_retain_gpu_model(phase, error_type):
    refs = []
    def fail_with_chain():
        try:
            raise ValueError("inner error")
        except ValueError as inner:
            if error_type is HTTPException:
                raise HTTPException(400, "model rejected") from inner
            raise error_type("model failed") from inner
    class FailedModel(FakeModel):
        def generate_voice_design(self, **kwargs):
            fail_with_chain()
    def loader(key):
        if key[0] == "VoiceDesign":
            model = FailedModel(key, [])
            refs.append(weakref.ref(model))
            if phase == "load":
                fail_with_chain()
            return model
        assert refs[-1]() is None, "failed model retained by traceback or exception chain"
        return FakeModel(key, [])
    rt = FusionRuntime(loader=loader)
    rt.start()
    try:
        with pytest.raises(error_type) as retained:
            rt.synthesize(text="hello", mode="voice_design", instruct="warm", postprocess=False)
        assert retained.value.__context__ is None
        assert retained.value.__cause__ is None
        if error_type is HTTPException:
            assert retained.value.status_code == 400
        rt.synthesize(text="hello", mode="custom_voice", speaker="Ryan", postprocess=False)
    finally:
        rt.shutdown()
