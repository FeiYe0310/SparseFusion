#!/usr/bin/env bash
set -euo pipefail

# 可选：使用现有代理变量
# export https_proxy=...
# export http_proxy=...

echo "Installing BFCL CLI from Qwen3-Coder subdirectory..."
pip install "git+https://github.com/QwenLM/Qwen3-Coder.git#subdirectory=qwencoder-eval/tool_calling_eval/berkeley-function-call-leaderboard"

echo "Verifying bfcl entrypoint..."
if bfcl --help >/dev/null 2>&1; then
  echo "bfcl is installed and available in PATH."
else
  echo "bfcl entrypoint not found in PATH. Trying module execution..."
  python -m bfcl_eval --help >/dev/null 2>&1 && echo "bfcl_eval module available." || {
    echo "ERROR: bfcl not available. Check Python environment and PATH." >&2
    exit 1
  }
fi

echo "Done."




