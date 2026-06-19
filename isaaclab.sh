#!/usr/bin/env bash

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

print_help() {
    cat <<'EOF'
usage: ./isaaclab.sh -p <script.py> [args...]

FlexiTac launcher. This repository no longer vendors IsaacLab source; install
official IsaacLab in the active Isaac/conda environment.

options:
  -h, --help      Show this help.
  -p, --python    Run a script with the active conda/venv Python, Isaac Sim
                  python.sh, or system python fallback.
EOF
}

extract_python_exe() {
    if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
        echo "${CONDA_PREFIX}/bin/python"
        return
    fi

    if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
        echo "${VIRTUAL_ENV}/bin/python"
        return
    fi

    if [[ -x "${REPO_ROOT}/_isaac_sim/python.sh" ]]; then
        echo "${REPO_ROOT}/_isaac_sim/python.sh"
        return
    fi

    if command -v python3 >/dev/null 2>&1; then
        command -v python3
        return
    fi

    command -v python
}

extend_isaaclab_pythonpath() {
    local python_exe="$1"
    local isaaclab_source_path

    isaaclab_source_path="$("${python_exe}" - <<'PY' 2>/dev/null || true
import importlib.util
import os

spec = importlib.util.find_spec("isaaclab")
if spec is None or spec.origin is None:
    raise SystemExit

package_dir = os.path.dirname(os.path.abspath(spec.origin))
candidates = [
    os.path.join(package_dir, "source", "isaaclab"),
    os.path.dirname(package_dir),
]
for path in candidates:
    if os.path.exists(os.path.join(path, "isaaclab", "assets")):
        print(path)
        break
PY
)"

    if [[ -n "${isaaclab_source_path}" ]]; then
        export PYTHONPATH="${isaaclab_source_path}:${PYTHONPATH:-}"
    fi
}

if [[ $# -eq 0 ]]; then
    print_help
    exit 0
fi

case "$1" in
    -h|--help)
        print_help
        ;;
    -p|--python)
        shift
        if [[ $# -eq 0 ]]; then
            echo "[ERROR] Missing script path after -p/--python." >&2
            exit 1
        fi
        python_exe="$(extract_python_exe)"
        if [[ -z "${python_exe}" ]]; then
            echo "[ERROR] Could not find a Python executable." >&2
            exit 1
        fi
        extend_isaaclab_pythonpath "${python_exe}"
        echo "[INFO] Using python from: ${python_exe}"
        "${python_exe}" "$@"
        ;;
    *)
        echo "[ERROR] Unsupported option: $1" >&2
        echo "[ERROR] This trimmed repository only supports './isaaclab.sh -p ...'." >&2
        exit 1
        ;;
esac
