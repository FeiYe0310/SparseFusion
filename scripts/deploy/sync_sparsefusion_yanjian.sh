#!/usr/bin/env bash
# Sparse_fusion two-way sync via Mutagen + SSH config alias
# Specific configuration that calls the general sync script

set -euo pipefail

# ---- Config ----
SESSION_NAME="sparse-fusion"                           # no underscores (Mutagen rule)
REMOTE_HOST_ALIAS="${REMOTE_HOST_ALIAS:-h.pjlab.org.cn}"
REMOTE_USER="${REMOTE_USER:-yanjian.zhangyanjian-p.ailab-formalverification.ws}"
REMOTE_PATH="/mnt/shared-storage-user/zhangyanjian-p/SparseFusion"
LOCAL_DIR="${LOCAL_DIR:-$(pwd)}"
STATE_ROOT="${STATE_ROOT:-${LOCAL_DIR%/}/.sparsefusion-sync}"

# Directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Call the general sync script with specific parameters
"${SCRIPT_DIR}/sync_general_yanjian.sh" \
  --session-name "${SESSION_NAME}" \
  --remote-host "${REMOTE_HOST_ALIAS}" \
  --remote-user "${REMOTE_USER}" \
  --remote-path "${REMOTE_PATH}" \
  --local-dir "${LOCAL_DIR}" \
  --state-root "${STATE_ROOT}" \
  "$@"
