#!/usr/bin/env python3
"""
MBPP 轮转（PyTorch→JAX→PyTorch）对比测试

使用环境变量提供路径：
  - MBPP_MODEL_PATH: 模型目录（如 /mnt/.../Qwen2.5-Coder-0.5B-Instruct）
  - MBPP_DATA_PATH:  本地 HuggingFace datasets 目录（如 /mnt/.../mbpp_hf）
可选：
  - MBPP_SUBSET: 评估子集大小（默认 10）
  - MBPP_BATCH:  批大小（默认 1）
"""

import os
import pytest
import torch

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover
    AutoModelForCausalLM = None
    AutoTokenizer = None

from torch.utils.data import DataLoader, Subset

from helper_fn import (
    pytorch_to_jax_flattened,
    jax_flattened_to_pytorch_model,
)
from utils.eval_utils import MBPPDataset, mbpp_collate_fn


def _clean_code_block(text: str) -> str:
    import re
    s = text.strip()
    m = re.search(r"```python\n([\s\S]*?)\n```", s, re.IGNORECASE)
    if not m:
        m = re.search(r"```\n([\s\S]*?)\n```", s)
    if m:
        return m.group(1).strip()
    m = re.search(r"def\s+\w+\s*\(.*", s)
    if m:
        return s[m.start():].strip()
    return s

def _parse_first_def_name(text: str) -> str | None:
    m = re.search(r"^\s*def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", text, re.MULTILINE)
    return m.group(1) if m else None


def _safe_execute_code(code: str, tests: list, setup_code: str = "", timeout: int = 10) -> bool:
    import subprocess
    import tempfile
    import uuid

    program_parts = []
    if setup_code:
        program_parts.append(setup_code)
    program_parts.append(code)
    program_parts.extend(tests)
    program_parts.append("print('__MBPP_ALL_TESTS_PASSED__')")
    program = "\n".join(program_parts)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, f"{uuid.uuid4().hex}.py")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(program)
            result = subprocess.run(
                ["python3", filepath],
                capture_output=True,
                text=True,
                timeout=timeout,
                env={"PYTHONDONTWRITEBYTECODE": "1"},
            )
            return ("__MBPP_ALL_TESTS_PASSED__" in (result.stdout or "")) and (result.returncode == 0)
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def _load_model_and_tokenizer(model_path: str, device: torch.device):
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # CPU: float32, CUDA: bfloat16（节省显存）
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, trust_remote_code=True
    ).to(device).eval()
    return model, tok


def _evaluate_mbpp(model, tokenizer, mbpp_path: str, batch_size: int, subset: int | None, device: torch.device) -> tuple[int, int]:
    ds = MBPPDataset(mbpp_path, tokenizer)
    eval_ds = Subset(ds, list(range(min(subset, len(ds))))) if subset is not None else ds
    loader = DataLoader(
        eval_ds, batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=lambda b: mbpp_collate_fn(b, tokenizer),
    )

    passed = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            tests_list = batch["test_list"]
            setup_codes = batch["test_setup_code"]
            imports_codes = batch.get("test_imports", [""]*len(setup_codes))
            ref_codes = batch.get("reference_code", [""]*len(setup_codes))

            gen_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            gen_texts = tokenizer.batch_decode(
                gen_ids[:, input_ids.shape[1]:], skip_special_tokens=True
            )

            for gen, tests, setup, imports_code, ref_code in zip(gen_texts, tests_list, setup_codes, imports_codes, ref_codes):
                code = _clean_code_block(gen)
                expected = _parse_first_def_name(ref_code or "")
                got = _parse_first_def_name(code)
                alias_line = f"{expected} = {got}" if expected and got and expected != got else ""
                combined_setup = "\n".join([x for x in [imports_code or "", setup or "", alias_line] if x])
                ok = _safe_execute_code(code, tests, combined_setup)
                passed += 1 if ok else 0
                total += 1
    return passed, total


@pytest.mark.skipif(AutoModelForCausalLM is None or AutoTokenizer is None, reason="transformers required")
def test_mbpp_roundtrip_on_qwen():
    model_path = os.environ.get("MBPP_MODEL_PATH")
    data_path = os.environ.get("MBPP_DATA_PATH")
    if not model_path or not data_path:
        pytest.skip("MBPP_MODEL_PATH and MBPP_DATA_PATH must be set")

    subset = int(os.environ.get("MBPP_SUBSET", "10"))
    batch_size = int(os.environ.get("MBPP_BATCH", "1"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 原模型推理
    model_orig, tokenizer = _load_model_and_tokenizer(model_path, device)
    orig_passed, orig_total = _evaluate_mbpp(model_orig, tokenizer, data_path, batch_size, subset, device)
    assert orig_total > 0, "MBPP数据为空或未正确加载"

    # 扁平化到JAX再恢复
    flat_params, param_shapes, _ = pytorch_to_jax_flattened(model_orig)
    model_restored, _ = _load_model_and_tokenizer(model_path, device)
    with torch.no_grad():
        for p in model_restored.parameters():
            p.zero_()
    _ = jax_flattened_to_pytorch_model(flat_params, model_restored, param_shapes)
    model_restored.eval()

    rt_passed, rt_total = _evaluate_mbpp(model_restored, tokenizer, data_path, batch_size, subset, device)

    # 断言：往返后通过率与原始相差不超过10%（允许少量数值差异）
    orig_acc = orig_passed / max(1, orig_total)
    rt_acc = rt_passed / max(1, rt_total)
    assert abs(rt_acc - orig_acc) <= 0.10, f"Roundtrip acc delta too large: orig={orig_acc:.3f}, rt={rt_acc:.3f}"


