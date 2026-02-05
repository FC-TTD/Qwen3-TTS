import sys
from pathlib import Path

# 直接把函数定义拷贝过来，进行纯逻辑验证，不依赖任何第三方库
def _infer_speaker_name(input_path: str) -> str | None:
    """从 UI 输入路径中推断 speaker 名称。"""
    try:
        if not input_path or not str(input_path).strip():
            return None

        # 模拟 APP_DATA_DIR / APP_LORA_DIR 逻辑，但在测试中我们主要看 Path.parts 表现
        p = Path(str(input_path).strip().replace("\\", "/"))
        parts = p.parts

        # 1. 检查是否在 /app/data/<speaker> 下
        try:
            # parts 可能类似 ('/', 'app', 'data', 'afu')
            # 或者在 Windows 上的 ('C:', 'app', 'data', 'afu')
            if "data" in parts:
                data_idx = parts.index("data")
                if data_idx > 0 and parts[data_idx-1].lower() in ["app", "c:", "d:"] and len(parts) > data_idx + 1:
                    spk = parts[data_idx + 1]
                    if spk.endswith(".qwen.jsonl"):
                        spk = spk[:-len(".qwen.jsonl")]
                    return spk
                # 兼容相对路径 data/afu
                elif data_idx == 0 and len(parts) > 1:
                    return parts[1]
        except (ValueError, IndexError):
            pass

        # 2. 检查是否在 /app/lora/<speaker> 下
        try:
            if "lora" in parts:
                lora_idx = parts.index("lora")
                if lora_idx > 0 and parts[lora_idx-1].lower() in ["app", "c:", "d:"] and len(parts) > lora_idx + 1:
                    return parts[lora_idx + 1]
        except (ValueError, IndexError):
            pass

        # 3. 检查文件名后缀
        name = p.name
        if name.endswith(".qwen.jsonl"):
            return name[:-len(".qwen.jsonl")] or None
        if name.endswith(".prepared.jsonl"):
            return name[:-len(".prepared.jsonl")] or None
        if name.endswith(".jsonl"):
            return p.stem or None

        # 4. 兜底：取最后一级目录
        if p.name and p.name not in ["app", "data", "lora", "/", "C:", "D:"]:
            return p.name
        
        return None
    except Exception:
        return None

def test_speaker_inference():
    test_cases = [
        # 标准数据目录
        ("/app/data/afu", "afu"),
        ("/app/data/afu/", "afu"),
        ("/app/data/afu.qwen.jsonl", "afu"),
        
        # 标准训练/输出目录
        ("/app/lora/afu", "afu"),
        ("/app/lora/afu/", "afu"),
        ("/app/lora/afu/checkpoint-100", "afu"),
        
        # Windows 风格路径
        (r"C:\app\data\afu", "afu"),
        ("app/data/afu", "afu"),
        ("data/afu", "afu"),
        
        # 特殊后缀
        ("my_voice.qwen.jsonl", "my_voice"),
        ("/tmp/someone.prepared.jsonl", "someone"),
        ("custom.jsonl", "custom"),
        
        # 兜底逻辑：取最后一级目录
        ("/data/test_user", "test_user"),
        ("simple_name", "simple_name"),
        
        # 边界情况
        ("", None),
        ("   ", None),
        ("/app/data", None),
        ("/app/lora/", None),
        ("/", None),
    ]

    print("\n开始验证 _infer_speaker_name 逻辑...")
    passed = 0
    failed = 0
    
    for input_path, expected in test_cases:
        actual = _infer_speaker_name(input_path)
        if actual == expected:
            print(f"OK [通过] 输入: '{input_path}' -> 推断: '{actual}'")
            passed += 1
        else:
            print(f"FAIL [失败] 输入: '{input_path}' | 期望: '{expected}' | 实际: '{actual}'")
            failed += 1
            
    print(f"\n测试结果: 通过 {passed}, 失败 {failed}")
    return failed == 0

if __name__ == "__main__":
    success = test_speaker_inference()
    sys.exit(0 if success else 1)
