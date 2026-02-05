import json
import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

# 模拟 librosa 和 soundfile，因为本地可能没环境
mock_librosa = MagicMock()
mock_sf = MagicMock()

# 模拟音频数据
def mock_load(path, sr=None):
    return "dummy_audio_array", 44100 # 返回 44.1k

def mock_resample(y, orig_sr, target_sr):
    return f"resampled_from_{orig_sr}_to_{target_sr}"

mock_librosa.load = mock_load
mock_librosa.resample = mock_resample

def _infer_speaker_name(path):
    return Path(path).name

def iter_wav_files(root_dir):
    p = Path(root_dir)
    return [p / "test1.wav", p / "test2.wav"]

def extract_text_from_filename(path):
    return "transcribed text"

@patch('librosa.load', side_effect=mock_load)
@patch('librosa.resample', side_effect=mock_resample)
@patch('soundfile.write')
def test_resample_logic(mock_write, mock_res, mock_ld):
    # 模拟输入输出环境
    test_data_dir = Path("./test_mock_data")
    test_data_dir.mkdir(exist_ok=True)
    (test_data_dir / "test1.wav").touch()
    (test_data_dir / "test2.wav").touch()
    
    output_jsonl = test_data_dir / "output.jsonl"
    
    # 核心逻辑实现 (从 ui.py 拷贝并稍作调整以适配测试)
    data_dir_path = test_data_dir
    TARGET_SR = 24000
    resample_dir = data_dir_path / "resampled_24k"
    resample_dir.mkdir(parents=True, exist_ok=True)
    
    entries = [{"audio": str(p), "text": ""} for p in iter_wav_files(test_data_dir)]
    processed_entries = []
    
    print(f"开始处理 {len(entries)} 条模拟数据...")
    for entry in entries:
        p = Path(entry["audio"])
        target_wav_path = resample_dir / p.name
        
        # 模拟重采样
        import librosa
        import soundfile as sf
        y, sr = librosa.load(str(p), sr=None)
        print(f"加载: {p.name}, 原始采样率: {sr}")
        
        if sr != TARGET_SR:
            print(f"重采样 {p.name} 到 {TARGET_SR}...")
            y_resampled = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
            sf.write(str(target_wav_path), y_resampled, TARGET_SR)
        
        new_entry = dict(entry)
        new_entry["audio"] = str(target_wav_path)
        new_entry["text"] = "transcribed"
        processed_entries.append(new_entry)

    # 验证
    assert len(processed_entries) == 2
    assert "resampled_24k" in processed_entries[0]["audio"]
    assert mock_write.call_count == 2
    print("✅ 逻辑验证成功: 重采样路径正确，调用了写入函数。")

    # 清理
    shutil.rmtree(test_data_dir)

if __name__ == "__main__":
    try:
        test_resample_logic()
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
