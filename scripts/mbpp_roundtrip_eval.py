#!/usr/bin/env python3
import os
import argparse
import json
import random
import torch
import sys

# Ensure JAX runs on CPU for flatten/restore
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

from transformers import AutoModelForCausalLM, AutoTokenizer

from torch.utils.data import DataLoader, Subset

# Add project root to sys.path so we can import helper modules when running from scripts/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from helper_fn import (
    pytorch_to_jax_flattened,
    jax_flattened_to_pytorch_model,
)
from mbpp_data_utils import MBPPDataset, mbpp_collate_fn


def safe_execute_code(code: str, tests: list, setup_code: str = "", timeout: int = 10) -> bool:
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

            success = (
                "__MBPP_ALL_TESTS_PASSED__" in (result.stdout or "") and result.returncode == 0
            )
            return success
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def clean_code_block(text: str) -> str:
    s = text.strip()
    if s.startswith("```python"):
        s = s[len("```python"):].strip()
    if s.startswith("```"):
        s = s[3:].strip()
    if s.endswith("```"):
        s = s[:-3].strip()
    return s


def load_model_and_tokenizer(model_path: str, device: torch.device):
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # For decoder-only generate, left padding is safer
    tok.padding_side = "left"

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float32 if device.type == "cpu" else torch.bfloat16,
            trust_remote_code=True,
        )
    except Exception:
        # Fallback in case class name differs
        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True
        )

    model.to(device)
    model.eval()
    return model, tok


def evaluate_mbpp_with_model(model, tokenizer, mbpp_path: str, batch_size: int, subset: int | None, device: torch.device) -> dict:
    ds = MBPPDataset(mbpp_path, tokenizer)
    if subset is not None and subset < len(ds):
        random.seed(0)
        indices = random.sample(range(len(ds)), subset)
        eval_ds = Subset(ds, indices)
    else:
        eval_ds = ds

    loader = DataLoader(
        eval_ds, batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=lambda b: mbpp_collate_fn(b, tokenizer),
    )

    passed = 0
    total = 0
    samples = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            tests_list = batch["test_list"]
            setup_codes = batch["test_setup_code"]
            prompts = batch["prompts"]

            gen_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            gen_texts = tokenizer.batch_decode(gen_ids[:, input_ids.shape[1]:], skip_special_tokens=True)

            for prompt, gen, tests, setup in zip(prompts, gen_texts, tests_list, setup_codes):
                code = clean_code_block(gen)
                ok = safe_execute_code(code, tests, setup)
                passed += 1 if ok else 0
                total += 1
                if len(samples) < 3:
                    samples.append({
                        "prompt": prompt[:200],
                        "generated": gen[:300],
                        "passed": ok,
                    })

    return {"passed": passed, "total": total, "acc": (passed / total if total else 0.0), "samples": samples}


def main():
    parser = argparse.ArgumentParser(description="MBPP roundtrip eval (PyTorch -> JAX -> PyTorch)")
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--mbpp_path", required=True, type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--subset", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load original model/tokenizer
    model_orig, tokenizer = load_model_and_tokenizer(args.model_path, device)

    # 2) Baseline eval
    print("\n=== Baseline (original parameters) ===")
    baseline = evaluate_mbpp_with_model(model_orig, tokenizer, args.mbpp_path, args.batch_size, args.subset, device)
    print(json.dumps({k: (v if k != "samples" else v) for k, v in baseline.items()}, ensure_ascii=False, indent=2))

    # 3) Flatten to JAX then restore into a fresh skeleton
    print("\nFlattening to JAX and restoring into a fresh skeleton...")
    flat_params, param_shapes, _ = pytorch_to_jax_flattened(model_orig)

    # Reload a fresh skeleton and zero it before restore
    model_skel, _ = load_model_and_tokenizer(args.model_path, device)
    with torch.no_grad():
        for p in model_skel.parameters():
            p.zero_()
    _ = jax_flattened_to_pytorch_model(flat_params, model_skel, param_shapes)
    model_skel.eval()

    # 4) Roundtrip eval
    print("\n=== After roundtrip (restored parameters) ===")
    roundtrip = evaluate_mbpp_with_model(model_skel, tokenizer, args.mbpp_path, args.batch_size, args.subset, device)
    print(json.dumps({k: (v if k != "samples" else v) for k, v in roundtrip.items()}, ensure_ascii=False, indent=2))

    # 5) Summary
    print("\n=== Summary ===")
    print(json.dumps({
        "baseline_acc": baseline["acc"],
        "roundtrip_acc": roundtrip["acc"],
        "delta": roundtrip["acc"] - baseline["acc"],
        "baseline_total": baseline["total"],
        "roundtrip_total": roundtrip["total"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


