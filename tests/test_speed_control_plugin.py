import os

import numpy as np

from ttd_fastapi_utils.speed_control import apply_speed_to_wav_list, time_stretch_wav


def _sine(sr: int = 48000, freq: float = 440.0, dur: float = 0.25) -> np.ndarray:
    t = np.arange(int(sr * dur), dtype=np.float32) / float(sr)
    return (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_time_stretch_speed_1_passthrough() -> None:
    wav = _sine()
    out = time_stretch_wav(wav, 48000, 1.0, allow_passthrough_on_failure=True)
    assert out.shape == wav.shape
    assert np.allclose(out, wav)


def test_bypass_env_passthrough(monkeypatch) -> None:
    monkeypatch.setenv("TTD_SPEED_CONTROL_BYPASS_SOX", "1")

    wav = _sine(dur=0.3)
    out = time_stretch_wav(wav, 48000, 1.2, allow_passthrough_on_failure=True)
    assert out.shape == wav.shape
    assert np.allclose(out, wav)


def test_apply_speed_to_wav_list_bypass_env(monkeypatch) -> None:
    monkeypatch.setenv("TTD_SPEED_CONTROL_BYPASS_SOX", "1")

    wavs = [_sine(dur=0.2), _sine(dur=0.25)]
    outs = apply_speed_to_wav_list(wavs, 48000, 0.8, allow_passthrough_on_failure=True)
    assert len(outs) == len(wavs)
    for o, w in zip(outs, wavs):
        assert o.shape == w.shape
        assert np.allclose(o, w)
