#!/usr/bin/env python3
import os
import argparse
import json
import random
import torch
import sys
import re

# Ensure JAX runs on CPU for flatten/restore
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

from transformers import AutoModelForCausalLM, AutoTokenizer

from torch.utils.data import DataLoader, Subset
from datasets import load_from_disk

# Add project root to sys.path so we can import helper modules when running from scripts/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from helper_fn import (
    pytorch_to_jax_flattened,
    jax_flattened_to_pytorch_model,
)
from utils.eval_utils import MBPPDataset, mbpp_collate_fn


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
    # 1) Try to extract the first fenced code block
    m = re.search(r"```python\n([\s\S]*?)\n```", s, re.IGNORECASE)
    if not m:
        m = re.search(r"```\n([\s\S]*?)\n```", s)
    if m:
        return m.group(1).strip()
    # 2) Fallback: take from the first 'def ' to the end
    m = re.search(r"def\s+\w+\s*\(.*", s)
    if m:
        return s[m.start():].strip()
    return s


def parse_first_def_name(text: str) -> str | None:
    m = re.search(r"^\s*def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", text, re.MULTILINE)
    return m.group(1) if m else None


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


def evaluate_mbpp_with_model(
    model,
    tokenizer,
    mbpp_path: str,
    batch_size: int,
    subset: int | None,
    device: torch.device,
    return_details: bool = False,
    max_details: int | None = None,
    use_three_shot: bool = False,
    few_shot_k: int = 3,
    few_shot_split: str = "train",
    use_qwen_chat: bool = False,
) -> dict:
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
    # Prepare exemplar pool for few-shot, if enabled
    exemplar_pool: list[dict] | None = None
    if use_three_shot and few_shot_k > 0:
        try:
            ds_all = load_from_disk(mbpp_path)
            if few_shot_split in ds_all and len(ds_all[few_shot_split]) > 0:
                exemplar_pool = list(ds_all[few_shot_split])
            else:
                # fallback to train/validation/test order
                for sp in ("train", "validation", "test"):
                    if sp in ds_all and len(ds_all[sp]) > 0:
                        exemplar_pool = list(ds_all[sp])
                        break
        except Exception:
            exemplar_pool = None

    def build_few_shot_prompt(cur_prompt: str) -> str:
        if not exemplar_pool:
            return cur_prompt
        rng = random.Random(0)
        examples = rng.sample(exemplar_pool, k=min(few_shot_k, len(exemplar_pool)))
        parts = []
        for i, ex in enumerate(examples, 1):
            ex_p = ex.get("prompt", "")
            ex_c = ex.get("code", "")
            parts.append(f"### Example {i}\nProblem: {ex_p}\nSolution:\n{ex_c}\n")
        parts.append(f"### Task\nProblem: {cur_prompt}\nSolution:")
        return "\n\n".join(parts)

    def build_qwen_chat_messages(cur_prompt: str, expected_name: str | None) -> list[dict]:
        messages: list[dict] = []
        # System instruction: strict output only function body
        system_text = (
            "You are a helpful coding assistant. Return ONLY valid Python function code. "
            "Do not include explanations, code fences, examples, or main blocks. "
            "If a specific function name is required, you MUST use that name."
        )
        messages.append({"role": "system", "content": system_text})

        if exemplar_pool:
            rng = random.Random(0)
            examples = rng.sample(exemplar_pool, k=min(few_shot_k, len(exemplar_pool)))
            for ex in examples:
                ex_p = ex.get("prompt", "")
                ex_c = ex.get("code", "")
                messages.append({"role": "user", "content": f"Problem: {ex_p}\nReturn only the function implementation."})
                messages.append({"role": "assistant", "content": ex_c})

        # Current task
        requirement = "Return only the function implementation."
        if expected_name:
            requirement += f" Function name MUST be `{expected_name}`."
        messages.append({"role": "user", "content": f"Problem: {cur_prompt}\n{requirement}"})
        return messages


    passed = 0
    total = 0
    samples = []

    with torch.no_grad():
        for batch in loader:            # Gather batch metadata first (needed by prompt builders)
            tests_list = batch["test_list"]
            setup_codes = batch["test_setup_code"]
            imports_codes = batch.get("test_imports", [""] * len(setup_codes))
            ref_codes = batch.get("reference_code", [""] * len(setup_codes))
            prompts = batch["prompts"]

            # Optionally rebuild inputs with few-shot prompts or Qwen chat template
            if (use_three_shot and few_shot_k > 0) or use_qwen_chat:
                texts: list[str] = []
                for p, ref in zip(prompts, ref_codes):
                    expected = parse_first_def_name(ref or "")
                    if use_qwen_chat and hasattr(tokenizer, "apply_chat_template"):
                        msgs = build_qwen_chat_messages(p, expected)
                        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                    else:
                        text = build_few_shot_prompt(p) if (use_three_shot and few_shot_k > 0) else p
                    texts.append(text)
                enc = tokenizer(texts, padding=True, truncation=True, max_length=1024, return_tensors="pt")
                input_ids = enc["input_ids"].to(device)
                attention_mask = enc["attention_mask"].to(device)
            else:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

            gen_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            gen_texts = tokenizer.batch_decode(gen_ids[:, input_ids.shape[1]:], skip_special_tokens=True)

            for prompt, gen, tests, setup, imports_code, ref_code in zip(
                batch["prompts"], gen_texts, tests_list, setup_codes, imports_codes, ref_codes
            ):
                code = clean_code_block(gen)
                # Function name aliasing
                expected_name = parse_first_def_name(ref_code or "")
                gen_name = parse_first_def_name(code)
                alias_line = ""
                if expected_name and gen_name and expected_name != gen_name:
                    alias_line = f"{expected_name} = {gen_name}"
                # Combine imports + setup + alias
                combined_setup = "\n".join([x for x in [imports_code or "", setup or "", alias_line] if x])
                ok = safe_execute_code(code, tests, combined_setup)
                passed += 1 if ok else 0
                total += 1
                if return_details:
                    if (max_details is None) or (len(samples) < max_details):
                        samples.append({
                            "prompt": prompt,
                            "generated": gen,
                            "cleaned": code,
                            "expected_name": expected_name,
                            "generated_name": gen_name,
                            "passed": ok,
                        })

    return {"passed": passed, "total": total, "acc": (passed / total if total else 0.0), "samples": samples}


def main():
    parser = argparse.ArgumentParser(description="MBPP roundtrip eval (PyTorch -> JAX -> PyTorch)")
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--mbpp_path", required=True, type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--subset", type=int, default=20)
    parser.add_argument("--print_n", type=int, default=0, help="Print first N detailed samples for baseline and roundtrip")
    parser.add_argument("--dump_path", type=str, default=None, help="Optional path to dump all detailed results as JSON")
    parser.add_argument("--three_shot", action="store_true", help="Enable 3-shot (few-shot) prompts for MBPP")
    parser.add_argument("--few_shot_k", type=int, default=3, help="Number of exemplars for few-shot prompts")
    parser.add_argument("--few_shot_split", type=str, default="train", help="Split to sample exemplars from (train/validation/test)")
    parser.add_argument("--qwen_chat", action="store_true", help="Use Qwen chat template for prompts (recommended for Qwen)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load original model/tokenizer
    model_orig, tokenizer = load_model_and_tokenizer(args.model_path, device)

    # 2) Baseline eval
    print("\n=== Baseline (original parameters) ===")
    baseline = evaluate_mbpp_with_model(
        model_orig, tokenizer, args.mbpp_path, args.batch_size, args.subset, device,
        return_details=(args.print_n > 0 or args.dump_path is not None),
        max_details=(args.print_n if args.print_n > 0 else None),
        use_three_shot=args.three_shot, few_shot_k=args.few_shot_k, few_shot_split=args.few_shot_split,
        use_qwen_chat=args.qwen_chat,
    )
    print(json.dumps({k: (v if k != "samples" else (v[:args.print_n] if args.print_n > 0 else [])) for k, v in baseline.items()}, ensure_ascii=False, indent=2))

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
    roundtrip = evaluate_mbpp_with_model(
        model_skel, tokenizer, args.mbpp_path, args.batch_size, args.subset, device,
        return_details=(args.print_n > 0 or args.dump_path is not None),
        max_details=(args.print_n if args.print_n > 0 else None),
        use_three_shot=args.three_shot, few_shot_k=args.few_shot_k, few_shot_split=args.few_shot_split,
        use_qwen_chat=args.qwen_chat,
    )
    print(json.dumps({k: (v if k != "samples" else (v[:args.print_n] if args.print_n > 0 else [])) for k, v in roundtrip.items()}, ensure_ascii=False, indent=2))

    # 5) Summary
    print("\n=== Summary ===")
    print(json.dumps({
        "baseline_acc": baseline["acc"],
        "roundtrip_acc": roundtrip["acc"],
        "delta": roundtrip["acc"] - baseline["acc"],
        "baseline_total": baseline["total"],
        "roundtrip_total": roundtrip["total"],
    }, ensure_ascii=False, indent=2))

    # Optional dump
    if args.dump_path:
        out = {
            "baseline": baseline,
            "roundtrip": roundtrip,
        }
        with open(args.dump_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\nSaved detailed results to: {args.dump_path}")


if __name__ == "__main__":
    main()


